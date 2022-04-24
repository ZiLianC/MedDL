import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.convInit = nn.Conv3d(in_channel, 64, 3, stride=1, padding=1)
        self.down_layers = self.make_down_layers(3)
        self.up_layers = self.make_up_layers(3)
        self.out_layers=self.make_out_map(256,4)
        self.bottom = nn.Conv3d(512, 512, 3, stride=1, padding=1)
    
    def make_down_layers(self,down_layer_num):
        down_layers = nn.ModuleList()
        channels = 64
        for i in range(down_layer_num):
            down_layer = nn.Sequential(
            nn.Conv3d(channels, channels*2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(channels*2, channels*2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(channels*2)
            )
            down_layers.append(down_layer)
            channels = channels*2
        return down_layers
        
    
    def make_up_layers(self,up_layer_num):
        up_layers = nn.ModuleList()
        channels = 512
        for i in range(up_layer_num):
            up_layer = nn.Sequential(
            nn.Conv3d(channels, channels//2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(channels//2, channels//2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(channels//2))
            up_layers.append(up_layer)
            channels = channels//2
        return up_layers
        
        
    def make_out_map(self,feature_channel,up_layer_num):
        out_layers=nn.ModuleList()
        for i in range(up_layer_num):
            out_layer = nn.Sequential(
            nn.Conv3d(feature_channel,2, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear',align_corners=True),
            nn.Softmax(dim =1)
            )
            out_layers.append(out_layer)
            feature_channel = feature_channel//2
        return out_layers
        
        
    def encoder(self,x):
        x = self.convInit(x)
        down_x = []
        for i in range(len(self.down_layers)):
            x = self.down_layers[i](x)
            down_x.append(x)
            x = F.max_pool3d(x,2,2)
        x= self.bottom(x)
        return x, down_x
    
    def decoder(self,x,down_x):
        up_x=[]
        for i, up in enumerate(self.up_layers):
            x = F.interpolate(x,scale_factor=(2,2,2),mode='trilinear',align_corners=True) + down_x[i]
            x=up(x)
            up_x.append(x)
        return x,up_x
    
    
    def forward(self, x):
        x,down_x=self.encoder(x)
        down_x.reverse()
        x,up_features=self.decoder(x,down_x)
        up_outs=[]
        for i,up in enumerate(up_features):
            up_outs.append(self.out_layers[i](up))
        if self.training is True:
            return up_outs # last one is the highest stack
        else:
            return x
            

if __name__ =='__main__':
    device = torch.device('cuda')
    model=UNet().to(device)
    dummy=torch.randn(1, 1, 64,64,64).float().to(device)
    x=model(dummy)
    print(x[0].shape)
    
    
    
    
    
    