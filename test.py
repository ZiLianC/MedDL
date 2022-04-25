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
import torch.utils.data.distributed
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from monai.inferers import sliding_window_inference
from utils.data_utils import get_loader
from utils.valid_utils import dice
from utils.visualization import print_cut_samples,print_heatmap
from utils.slide_saliency_inference import sliding_saliency_inference
from monai.inferers import SaliencyInferer

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
# datas and models info.
parser.add_argument('--typeoftask', default='segmentation', type=str, help='type of DL task')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--pretrained_model_name', default='model_best_model.pth', type=str, help='pretrained model name')
parser.add_argument('--state_dict', action='store_true', help='Using state_dict style pretrained')
parser.add_argument('--data_dir', default='./dataset/', type=str, help='dataset directory')
parser.add_argument('--data_list', default='dataset_0.json', type=str, help='dataset json file')

#visualization
parser.add_argument('--save_img', default='test', type=str, help='image cut save file')

# inference info.
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')

# data preprocessing
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--workers', default=4, type=int, help='number of workers')

# miscellenous
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=2, type=int, help='number of output channels')
parser.add_argument('--distributed', action='store_true', help='distributed training models')
parser.add_argument('--saliency', action='store_true', help='Print Saliency Map')


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
    val_loader = get_loader(args)
    # load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init printing saliency map. Implemented by MONAI. CAM GRADCAM supported
    if args.saliency:
        sf=SaliencyInferer("CAM","conv_final.0")

    # load model
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    
    if args.state_dict: # you need to define model by hand. uh ohh.
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
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict["state_dict"])
            
    # load model
    model=torch.load(pretrained_pth)
    
    for index ,(name, param) in enumerate(model.named_parameters()):
        print( str(index) + " " +name)

    # for sliding window inference
    inf_size = [args.roi_x, args.roi_y, args.roi_x]

    model.eval()
    model.to(device)
    start_time = time.time()
    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))
            # sliding windows inference
            
            if args.saliency & args.typeoftask=='classification':
                heatmap = sf(val_inputs,model)
                path=img_name.split(".")[0]+args.pretrained_model_name.split(".")[0]+"heatmap"
                print_heatmap(val_inputs,val_labels,val_outputs,salient,path)
            
            else: val_outputs = sliding_window_inference(val_inputs,
                                                   (96, 96, 96),
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            
            if args.typeoftask == 'segmentation':
                #visualization
                path=img_name.split(".")[0]+args.pretrained_model_name.split(".")[0]+".png"
                print_cut_samples(val_inputs,val_labels,val_outputs,path)
                # postprocess softmax->argmax for segmentation.
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
                val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            
            dice_list_sub = []
            # calculate desired metric
            organ_Dice = dice(val_outputs[0] == 1, val_labels[0] == 1)
            dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
            torch.cuda.empty_cache()
        # ave. metric.
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

if __name__ == '__main__':
    main()