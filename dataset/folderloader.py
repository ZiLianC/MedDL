from torchvision import transforms, datasets
import torch
import os

def getfolderloader(path,batch_size):
    datasettrain=torchvision.datasets.ImageFolder(
                       os.path.join(path,"train"), transform=transforms.Compose([
                       transforms.Resize([96,96]),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]), target_transform=None, is_valid_file=None)
                   
    datasetvalid=torchvision.datasets.ImageFolder(
                       os.path.join(path,"valid"), transform=transforms.Compose([
                       transforms.Resize([96,96]),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]), target_transform=None, is_valid_file=None)
    train_loader = torch.utils.data.DataLoader(datasettrain, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Test dataset
    val_loader = torch.utils.data.DataLoader(datasetvalid, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader,val_loader