import os
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset as dataset
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np


from sklearn.preprocessing import OneHotEncoder

def read_img(path):
    img = Image.open(path)
    return img


class csvloader(torch.utils.data.Dataset):
    def __init__(self,path,split='train', transform=None):
        self.split=split
        self.path=path
        self.img_list,self.label_list = self.load_all_file(os.path.join(path,"annotation"))
        self.train_transforms = transforms.Compose([
                       transforms.Resize([96,96]),
                       transforms.RandomHorizontalFlip(0.3),
                       transforms.RandomRotation(90),
                       transforms.RandomVerticalFlip(0.3),
                       transforms.RandomEqualize(0.3),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                   ])
        self.test_transforms = transforms.Compose([
                       transforms.Resize([96,96]),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
      
                   ])

        

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img=read_img(os.path.join(self.path,"images",self.img_list[index]))
        if self.split=='train' :
            img=self.train_transforms(img)
            return img,self.label_list[index]
        else:
            img=self.test_transforms(img)
            return img,self.label_list[index]


    def load_all_file(self,dir_path):
        file_list = dir_path+"/"+self.split+".csv"
        with open(file_list, 'r') as f:#将data_dir和label_file路径拼
            lines =f.readlines()[1:]
            img_list=[]
            label_list=[]
            for l in lines:
                tokens = l.rstrip().split(',') #这里注意，从csv里直接读进来是有，的，用split        
                                             #将其去掉并分割
                jpg_path, label = tokens
                img_list.append(jpg_path)
                label_list.append(int(label))
        
        return img_list,label_list
        
def getcsvloader(path,batch_size):
    train_dst=csvloader(path,split='train')
    test_dst=csvloader(path,split='test')
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

