import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from monai.metrics import DiceMetric
from dataset import csvloader,suploader
from modelzoo.classification.ResNet50 import ResNet50
from modelzoo.classification.ConvNeXt import ConvNeXt
from monai.networks.nets import DenseNet121
from trainer_class import run_training
from scheduler.warmup_cosineannealing import warmup_CosineAnnealing
from functools import partial
from sklearn.metrics import accuracy_score
from loss.supcon import SupConLoss
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
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')

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
    main_worker(gpu=args.gpus, args=args)



def main_worker(gpu, args):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    # assign gpu number
    args.gpu = gpu
    
    # Segment voxel into small voxels for memory efficient learning. also for sliding window inference.
    inf_size = [args.roi_x, args.roi_y, args.roi_x]
    pretrained_dir = args.pretrained_dir
    
    # card settings
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    
    # for training mode
    args.test_mode = False
    
    # load dataset
    trainloader,testloader = suploader.getcsvloader("./data/problem2_datas",16)
    '''trainloader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=16, shuffle=True)
    testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1, shuffle=True)'''
    print(args.rank, ' gpu', args.gpu)
    
    # print batchsize
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)
        
    

    # DEFINE MODELS HERE.
    model=ResNet50(3,7,use_feature=False)
    #model=ConvNeXt(3,num_classes=7)
    #model=DenseNet121(spatial_dims=2, in_channels=3,
                   #out_channels=7)
    # DEFINE LOSS FUNC HERE.
    # using dice_loss+cross entrophy for loss function
    weight=[0.15,0.05,0.15,0.15,0.15,0.15,0.15]
    weight=torch.tensor(weight,dtype=torch.float).cuda()
    bce_loss = nn.CrossEntropyLoss(weight=weight,label_smoothing=0.1)
    con_loss =SupConLoss(temperature=0.07)
                            
    # param number
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    # counter
    best_acc = 0
    start_epoch = 0
    
    # model downstream to card
    model.cuda(args.gpu)
    
    # DEFINE OPTIMIZER HERE
    '''optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,weight_decay=2e-05)'''
    optimizer=torch.optim.SGD(model.parameters(),
                          lr=args.optim_lr,
                          momentum=0.9,
                          weight_decay=1e-4)
                                     
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.max_epochs)

    # push downstream for learning. combined all essentials.
    accuracy = run_training(model=model,
                            train_loader=trainloader,
                            val_loader=testloader,
                            optimizer=optimizer,
                            loss_func=bce_loss,
                            loss_con= con_loss,
                            acc_func=accuracy_score,
                            args=args,
                            model_inferer=None,
                            scheduler=scheduler,
                            start_epoch=start_epoch,
                            post_label=None,
                            post_pred=None)
    return accuracy

if __name__ == '__main__':
    main()
