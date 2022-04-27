import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlocks(nn.Module):
    def __init__(self, in_channel=1,in_filter=1,out_filter=2, stride=1,downsample=True):
        super(ResBlocks, self).__init__()
        self.BottleNeck=self.make_BottleNeck(in_channel,in_filter,out_filter,stride)
        self.is_downsample=downsample
        if self.is_downsample:
            self.downsample=nn.Sequential(
            nn.Conv2d(in_channel,out_filter, 1, stride=stride),
            nn.GroupNorm(8,out_filter))
        
    def make_BottleNeck(self,input_channel,in_filter,out_filter,stride):
        return nn.Sequential(
            nn.Conv2d(input_channel,in_filter, 1, stride=stride,bias=False),
            nn.GroupNorm(8,in_filter),
            nn.Conv2d(in_filter, in_filter, 3, stride=1, padding=1,bias=False),
            nn.GroupNorm(8,in_filter),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_filter,out_filter, 1, stride=1,bias=False),
            nn.GroupNorm(8,out_filter)
            )
            
    def forward(self,x):
        shortcut = x
        x=self.BottleNeck(x)
        if self.is_downsample:
            x=x+self.downsample(shortcut)
            return x
        return x+shortcut




class ResNet50(nn.Module):
    def __init__(self, in_channel=1,class_num=1000,use_feature=False):
        super(ResNet50, self).__init__()
        self.use_feature=use_feature
        self.input_conv = nn.Sequential(
        nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3),
        nn.GroupNorm(8,64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1=self._make_layer(64,64,256,3)
        self.layer2=self._make_layer(256,128,512,4,2)
        self.layer3=self._make_layer(512,256,1024,6,2)
        self.layer4=self._make_layer(1024,512,2048,3,2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 7))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    
    def _make_layer(self, in_channels,in_filters, out_filters, blocks_num, stride = 1):
        layers = []
        block_one = ResBlocks(in_channels,in_filters, out_filters,stride,downsample=True)
        layers.append(block_one)
        for i in range(1, blocks_num):
            layers.append(ResBlocks(out_filters, in_filters, out_filters, stride=1,downsample=False))
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x=self.input_conv(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        if self.use_feature:
            return x
        # out
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        #x = F.normalize(x,dim=1)
        return x
            

if __name__ =='__main__':
    device = torch.device('cuda')
    model=ResNet50(3,3,use_feature=True).to(device)
    dummy=torch.randn(1, 3, 64,64).float().to(device)
    x=model(dummy)
    print(x.shape)
    
    
    
    
    
    