import os
import torch
import h5py
from torch.utils.data import Dataset as dataset
from monai import transforms, data
from torch.utils.data import DataLoader

def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label


class LAHeart(torch.utils.data.Dataset):
    def __init__(self,path,split='train', transform=None):
        self.split=split
        self.path=path
        self.file_list = self.load_all_file(os.path.join(path,split))
        self.train_transforms = transforms.Compose(
        [
           transforms.AddChanneld(keys=["image", "label"]),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(48, 48, 48),
                pos=1,
                neg=1,
                num_samples=16,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=0.2,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=0.2,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=0.2,
                                 spatial_axis=2),
            transforms.RandRotate90d(
                keys=["image", "label"],
                prob=0.2,
                max_k=3,
            ),
            transforms.RandScaleIntensityd(keys="image",
                                           factors=0.1,
                                           prob=0.2),
            transforms.RandShiftIntensityd(keys="image",
                                           offsets=0.1,
                                           prob=0.1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
        self.test_transforms = transforms.Compose(
        
        [
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.ToTensord(keys=["image", "label"]),
        ])

        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img,lab=read_h5(os.path.join(self.path,self.split,self.file_list[index]))
        if self.split=='train':
            dic={"image":img,"label":lab}
            new_dic=self.train_transforms(dic)
            return new_dic[0]['image'],new_dic[0]['label']
        else:
            dic={"image":img,"label":lab}
            new_dic=self.test_transforms(dic)
            return new_dic['image'],new_dic['label']


    def load_all_file(self,dir_path):
        file_list = os.listdir(dir_path)
        file_numbers = len(file_list)
        file = []
        for h5_file,i in zip(file_list,range(file_numbers)):
            file.append(h5_file)
            
        return file
        
def geth5loader(path,batch_size):
    train_dst=LAHeart(path,split='train')
    test_dst=LAHeart(path,split='test')
    train_loader = DataLoader(
        train_dst, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dst, 
        batch_size=1, 
        shuffle=True,
        num_workers=0, 
        pin_memory=True
    )
    return train_loader,test_loader

