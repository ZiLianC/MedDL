import os
import torch
import h5py
from torchvision.transforms import transforms
from torch.utils.data import Dataset as dataset


def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label


class LAHeart(torch.utils.data.Dataset):
    def __init__(self,args,split='train', transform=None):
        self.args = args
        self.split=split
        self.file_list = self.load_all_file(os.join(args.dataset,split))
        self.train_transforms = transforms.Compose([transforms.()])
        self.test_transforms = transforms.Compose([transforms.()])

        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img,lab=read_h5(self.file_list[index])
        if self.split=='train':
            img,lab=self.train_transforms(img,lab)
            return img,lab
        else:
            img,lab=self.test_transforms(img,lab)
            return img,lab


    def load_all_file(self,dir_path):
        file_list = os.listdir(dir_path)
        file_numbers = len(file_list)
        file = []
        for h5_file,i in zip(file_list,range(file_numbers)):
            file.append(h5_file)
        return file


