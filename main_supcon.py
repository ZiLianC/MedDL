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
from trainer_supcon import run_training
from scheduler.warmup_cosineannealing import warmup_CosineAnnealing
from functools import partial
from sklearn.metrics import accuracy_score
from loss.supcon import SupConLoss
import argparse
from torchvision.models import convnext,resnet

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
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
# training & validation policy
parser.add_argument('--max_epochs', default=5000, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=2, type=int, help='number of batch size')
parser.add_argument('--val_every', default=10, type=int, help='validation frequency')
# cuda optimazation
parser.add_argument('--noamp', action='store_true', help='do NOT use amp for training')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
# optimizer policy
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')

# scheduler policy

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
    pretrained_dir = args.pretrained_dir
    
    # card settings
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    
    # for training mode
    args.test_mode = False
    
    # load dataset
    trainloader,testloader = suploader.getcsvloader("./data/problem2_datas",args.batch_size)
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
    #model=ResNet50(3,7,use_feature=False)
    resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
    model = resnet.resnet50(pretrained=True)
    model.fc = nn.Linear(2048,7)
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
