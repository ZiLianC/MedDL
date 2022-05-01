import os
import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from monai.inferers import sliding_window_inference
from monai.transforms import Compose
from monai.losses import DiceLoss, DiceCELoss
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete,Activations,Compose
from monai.metrics import DiceMetric,HausdorffDistanceMetric,SurfaceDistanceMetric
from torchmetrics import JaccardIndex
from dataset import h5Loader
from modelzoo.segmentation.segres import SegResNet
from trainer import run_training
from scheduler.warmup_cosineannealing import warmup_CosineAnnealing
from functools import partial
import argparse

parser = argparse.ArgumentParser(description='Template pipeline')

# load/save models & logs & weights &datasets
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='/dataset/dataset0/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_0.json', type=str, help='dataset json file')
parser.add_argument('--save', default='model.pth', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='UNETR_model_best_acc.pth', type=str, help='pretrained model name')
parser.add_argument('--save_checkpoint', action='store_true', help='save checkpoint during training')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--model_name', default='unetr', type=str, help='model name')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')

# training & validation policy
parser.add_argument('--max_epochs', default=5000, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=2, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=1, type=int, help='number of sliding window batch size')
parser.add_argument('--val_every', default=10, type=int, help='validation frequency')
parser.add_argument('--roi_x', default=48, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=48, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=48, type=int, help='roi size in z direction')

# cuda optimazation
parser.add_argument('--noamp', action='store_true', help='do NOT use amp for training')

# optimizer policy
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')

# scheduler policy
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=5, type=int, help='number of warmup epochs')

# distributed learning. NOTICE: models exported in this training method has minor difference when reloaded to test.
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')

# data preprocess & augmentation == USING MONAI METHOD. REFER TO https://docs.monai.io/en/stable/
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')

# Miscellous
parser.add_argument('--smooth_dr', default=1e-6, type=float, help='constant added to dice denominator to avoid nan')
parser.add_argument('--smooth_nr', default=0.0, type=float, help='constant added to dice numerator to avoid zero')
parser.add_argument('--gpus', type=int,default=0)


def main():
    args = parser.parse_args()
    # opt. for autograd
    args.amp = not args.noamp
    # log dir
    args.logdir = './runs/' + args.logdir
    # distributed learning backend & workplace init.
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    # single card learning just start.
    else:
        main_worker(gpu=args.gpus, args=args)



def main_worker(gpu, args):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    # assign gpu number
    args.gpu = gpu
    
    # Segment voxel into small voxels for memory efficient learning. also for sliding window inference.
    inf_size = [args.roi_x, args.roi_y, args.roi_x]
    pretrained_dir = args.pretrained_dir
    
    # init distributed learning (optional)
    if args.distributed:
        # start multiprocess for distributed learning
        torch.multiprocessing.set_start_method('fork', force=True)
        # set rank
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    # card settings
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    
    # for training mode
    args.test_mode = False
    
    # load dataset
    trainloader,testloader = h5Loader.geth5loader("./data/problem1_datas",args.batch_size)
    print(args.rank, ' gpu', args.gpu)
    
    # print batchsize
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)
        
    

    # DEFINE MODELS HERE.
    model=SegResNet(spatial_dims=3,
            in_channels=1,
            out_channels=2,
            upsample_mode="deconv",
            using_features=False)
        
        # procedure for resume learning & finetuning & transfer learning
    if args.resume_ckpt:
        # Transfer Learning loading pretrained weights
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
        model_state_dict = model.state_dict()
        # load corresponding layer weights
        state_dict = {k:v for k,v in model_dict.items() if k in model_state_dict.keys()}
        #del state_dict["out.conv.conv.weight"]
        #del state_dict["out.conv.conv.bias"]
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        print('Use pretrained weights')

    # DEFINE LOSS FUNC HERE.
    # using dice_loss+cross entrophy for loss function
    dice_loss = DiceCELoss(to_onehot_y=True,
                           softmax=True,
                           squared_pred=True,
                           smooth_nr=args.smooth_nr,
                           smooth_dr=args.smooth_dr)
                           
    # validation pipeline - metric & post process
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True,
                           to_onehot=2)
    acc_func=[]
    acc_func.append(DiceMetric(include_background=True,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=True))
    acc_func.append(JaccardIndex(num_classes=2))
    acc_func.append(SurfaceDistanceMetric(include_background=True,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=True))
    acc_func.append(HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=95, directed=False, reduction=MetricReduction.MEAN, get_not_nans=True))
    
    
    #using sliding window inference for memory efficient inference
    model_inferer = partial(sliding_window_inference,
                            roi_size=inf_size,
                            sw_batch_size=args.sw_batch_size,
                            predictor=model,
                            overlap=args.infer_overlap)
                            
    # param number
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    # counter
    best_acc = 0
    start_epoch = 0
    
    # model downstream to card
    model.cuda(args.gpu)
    
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == 'batch':
            # in distributed learning, MUST use sync_batchnorm!
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        # set parallel.
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          output_device=args.gpu,
                                                          find_unused_parameters=True)
    # DEFINE OPTIMIZER HERE
    if args.optim_name == 'adamw':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

    # DEFINE SCHEDULER HERE
    if args.lrschedule == 'warmup_cosine':
        scheduler = warmup_CosineAnnealing(optimizer,
                                                  warm_up_iter=args.warmup_epochs,
                                                  T_max=args.max_epochs)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    # push downstream for learning. combined all essentials.
    accuracy = run_training(model=model,
                            train_loader=trainloader,
                            val_loader=testloader,
                            optimizer=optimizer,
                            loss_func=dice_loss,
                            acc_func=acc_func,
                            args=args,
                            model_inferer=model_inferer,
                            scheduler=scheduler,
                            start_epoch=start_epoch,
                            post_label=post_label,
                            post_pred=post_pred)
    return accuracy

if __name__ == '__main__':
    main()
