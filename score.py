from monai.metrics import DiceMetric,SurfaceDistanceMetric
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete,Activations,Compose
from monai.data import decollate_batch
import os
import torch
import numpy as np
from monai.inferers import sliding_window_inference

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.utils.data.distributed
from monai.inferers import sliding_window_inference
from dataset import h5Loader
from utils.visualization import print_cut_samples,print_heatmap

from monai.inferers import SaliencyInferer

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='./dataset/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_0.json', type=str, help='dataset json file')
parser.add_argument('--save_img', default='test', type=str, help='image cut save file')
parser.add_argument('--pretrained_model_name', default='UNETR_model_best_acc.pth', type=str, help='pretrained model name')
parser.add_argument('--saved_checkpoint', default='ckpt', type=str, help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=16, type=int, help='feature size dimention')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=2, type=int, help='number of output channels')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=5, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=48, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=48, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=48, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')

def main():
    args = parser.parse_args()
    if args.distributed:
        dist.init_process_group(backend='nccl',
                                init_method='tcp://127.0.0.1:23456',
                                world_size=1,
                                rank=0)
    torch.cuda.empty_cache() 

    # enable test mode
    args.test_mode = True
    # load data
    trainloader,valloader = h5Loader.geth5loader("./data/problem1_datas",1)
    # load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    sf=SaliencyInferer("CAM","conv_final.0")
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model=torch.load(pretrained_pth)
    '''for index ,(name, param) in enumerate(model.named_parameters()):
        print( str(index) + " " +name)'''
    '''model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate)'''
    inf_size = [args.roi_x, args.roi_y, args.roi_x]

    '''model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    model.load_state_dict(model_dict["state_dict"])'''
    model.eval()
    model.to(device)
    start_time = time.time()
    dice_acc = DiceMetric(include_background=True,
    reduction=MetricReduction.MEAN)
    ASD = SurfaceDistanceMetric(include_background=True,
    reduction=MetricReduction.MEAN)
    # validation pipeline - metric & post process
    post_label = AsDiscrete(to_onehot=True,
                            n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True,
                           to_onehot=True,
                           n_classes=args.out_channels)
    with torch.no_grad():
        dice_list_case = []
        asd_list_case = []
        for i, batch in enumerate(valloader):
            torch.cuda.empty_cache()
            val_inputs, val_labels = batch
            val_inputs=val_inputs.cuda()
            val_labels=val_labels.cuda()
            img_name = i
            print("Inference on case {}".format(img_name))
            # sliding windows inference
            torch.cuda.empty_cache()
            val_outputs = sliding_window_inference(val_inputs,
                                                   (48, 48, 48),
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            path=str(i)+args.pretrained_model_name.split(".")[0]
            print(path)
            print_cut_samples(val_inputs,val_labels,val_outputs,path)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_list_sub = []
            asd_list_sub = []
            # calculate desired metric
            organ_Dice = dice_acc(y_pred=val_output_convert, y=val_labels_convert)
            organ_Dice=organ_Dice.detach().cpu().numpy()
            dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            organ_ASD = ASD(y_pred=val_output_convert, y=val_labels_convert)
            organ_ASD=organ_ASD.detach().cpu().numpy()
            asd_list_sub.append(organ_ASD)
            mean_ASD = np.mean(asd_list_sub)
            print("Mean Organ ASD: {}".format(mean_ASD))
            dice_list_case.append(mean_dice)
            asd_list_case.append(mean_ASD)
            torch.cuda.empty_cache()
        # ave. metric.
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
        print("Overall Mean ASD: {}".format(np.mean(asd_list_case)))

if __name__ == '__main__':
    main()