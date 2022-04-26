
# MedDL -- Implementations All In One

## Overview

This repository aims for fast cross-task, cross-model implementation framework with decoupled & flexible modules. It can give you a quick start. Templates are provided for basic pipelines such as training, testing, exporting models & results, and visualization. Minor accomondations are needed when using templates. Distributed learning and transfer learning workflows are also demonstrated in templates. Hope it can ultimately become the Timm in Medical DL.

I'm also build a model zoo of my own for reliable implementations of network when joining contest, and I'm trying to be a network architect:). Fork this if you like and welcome to contribute models. Issues and PRs are welcomed.

This framework is built with [Pytorch](https://pytorch.org/), [MONAI](https://monai.io/index.html), and inspiration from Prof. Li's assignments.



## Installing Dependencies

[Dependencies](./requirements.txt) can be installed using:

``` bash
pip install -r requirements.txt [-i https://pypi.tuna.tsinghua.edu.cn/simple/] (Optional)
```

Conda is recommended for a clean develop environment.

## Model Zoo

### Segmenatation [(View Trend)](https://paperswithcode.com/task/medical-image-segmentation)

- [x] 3D UNet [1] (with Deep Supervision [2])
- [ ] H-DenseUNet [3]
- [x] UNETR [4]
- [x] SegResNet (with VAE) [5] [#TODO]
- [x] TCFuse # Under trials

### Classification [(View Trend)](https://paperswithcode.com/task/image-classification)

- [x] PoolFormer [6]
- [x] ConvNeXt [7]
- [x] ResNet-50 [8] (ResNet Strikes Back[12])[Pytorch Blog](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#mixup-and-cutmix)
  <Regarding ISIC 2018, we resize the spatial resolution of the original image from 600×450 to 240×240, randomly crop a 224×224 region, and normalize the cropped region as the network input.

### Temporal Recoginition [(Related Field)](https://paperswithcode.com/task/weakly-supervised-temporal-action)

- [x] ResNet-50+LSTM
- [ ] Supposed to have a transformer based method.. [#TODO]

### Utils Blocks

- [x] Bottleneck


## Losses & Metrics

### Losses

- [x] Cross Entrophy
- [x] Dice Cross Entrophy
- [x] SupCon [10]
- [x] NCE
- [ ] AdaCon [11] [#TODO] [For regression use. The key is the adaptive margin. So how to implement the adaptive margin in discrete target is a problem.]
-  [Others](https://docs.monai.io/en/stable/losses.html)

### Metrics

- [x] DICE
- [x] ASD
- [ ] [Others](https://docs.monai.io/en/stable/metrics.html)

## Optmizers & Schedulers

### Optimizers

- Adam
- Adamw
- SGD

### Schedulers

- CosineAnnealing
- [x] WarmupCosineAnnealing

## Data Preparation

The [dataset](./dataset) directory implements Dataloader for:

- H5 Files
- .nii Files
- Classification Datasets in a directory style
- [#TODO] Loading Detection Datasets..

## Visualization

Use Tensorboard.

``` bash
tensorboard --logdir "./runs" --bind_all
```

## Usage

- For general pipeline, please refer to [Template](./Templates)
- For model implementation & prototype, please refer to [ModelZoo](./modelzoo) and follow the template's practice. Not sure available for all models but it's enough for a quick demo or validation.
- For loss & metric implementation, please refer to [Loss](./loss) and [Metric](./metric) respectively.
- For Visualization, please refer to [Visualization](./utils).



### Training

Use main.py for training/finetuning pipeline. Both 2D and 3D inputs are supported. However, careful examination should be made when dealing with dataloaders in case of tensor mismatch.

Note: In 3D voxel processing, sliding window inference is a default implementation. Image should be sampled when using ```Transforms``` and should pay attention to making the sample size correspond with inference input size.

### Testing

Use test.py for testing/vis. pipeline.



## References
[1] Çiçek, Özgün, et al. ‘3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation’. ArXiv:1606.06650 [Cs], June 2016. arXiv.org, http://arxiv.org/abs/1606.06650.

[2] Lee, Chen-Yu, et al. ‘Deeply-Supervised Nets’. ArXiv:1409.5185 [Cs, Stat], Sept. 2014. arXiv.org, http://arxiv.org/abs/1409.5185.

[3] Li, Xiaomeng, et al. ‘H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation From CT Volumes’. IEEE Transactions on Medical Imaging, vol. 37, no. 12, Dec. 2018, pp. 2663–74. DOI.org (Crossref), https://doi.org/10.1109/TMI.2018.2845918.

[4] Hatamizadeh, Ali, et al. ‘UNETR: Transformers for 3D Medical Image Segmentation’. ArXiv:2103.10504 [Cs, Eess], Oct. 2021. arXiv.org, http://arxiv.org/abs/2103.10504.

[5] Myronenko, Andriy. ‘3D MRI Brain Tumor Segmentation Using Autoencoder Regularization’. ArXiv:1810.11654 [Cs, q-Bio], Nov. 2018. arXiv.org, http://arxiv.org/abs/1810.11654.

[6] Yu, Weihao, et al. ‘MetaFormer Is Actually What You Need for Vision’. ArXiv:2111.11418 [Cs], Nov. 2021. arXiv.org, http://arxiv.org/abs/2111.11418.

[7] Liu, Zhuang, et al. ‘A ConvNet for the 2020s’. ArXiv:2201.03545 [Cs], Mar. 2022. arXiv.org, http://arxiv.org/abs/2201.03545.

[8] He, Kaiming, et al. ‘Deep Residual Learning for Image Recognition’. ArXiv:1512.03385 [Cs], Dec. 2015. arXiv.org, http://arxiv.org/abs/1512.03385.

[9] Tan, Mingxing, and Quoc V. Le. ‘EfficientNetV2: Smaller Models and Faster Training’. ArXiv:2104.00298 [Cs], June 2021. arXiv.org, http://arxiv.org/abs/2104.00298.

[10] Khosla, Prannay, et al. ‘Supervised Contrastive Learning’. ArXiv:2004.11362 [Cs, Stat], Mar. 2021. arXiv.org, http://arxiv.org/abs/2004.11362.

[11] Dai, Weihang, et al. ‘Adaptive Contrast for Image Regression in Computer-Aided Disease Assessment’. IEEE Transactions on Medical Imaging, 2021, pp. 1–1. DOI.org (Crossref), https://doi.org/10.1109/TMI.2021.3137854.

[12] Wightman, Ross, et al. ‘ResNet Strikes Back: An Improved Training Procedure in Timm’. ArXiv:2110.00476 [Cs], 1, Oct. 2021. arXiv.org, http://arxiv.org/abs/2110.00476.

