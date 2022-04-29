from modelzoo.classification.ResNet50 import ResNet50
#from ResNet50 import ResNet50
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext,resnet

class TRN(nn.Module):
    def __init__(self,in_channel,input_size,seq_size,class_num=1000):
        super(TRN, self).__init__()
        self.seq_size=seq_size
        self.input_size=input_size
        #self.feature_extractor=ResNet50(in_channel,class_num,use_feature=True)
        resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
        model = resnet.resnet50(pretrained=True)
        net_structure = list(model.children())
        print(net_structure)
        self.feature_extractor = nn.Sequential(*net_structure[:-1])
        self.lstm = nn.LSTM(2048, 2048,3, batch_first=True,bias=False)
        self.fc = nn.Linear(2048, 2048)
        self.out = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, class_num))
    
    def forward(self, x):
        #x.shape [batchsize,seqsize,3,inputsize,inputsize]
        B = x.size(0)
        x = x.view(B * self.seq_size, 3, self.input_size, self.input_size)# [batchsize,seqsize,3,inputsize,inputsize] -> [batchsize*seqsize,2048,2,2]
        x = self.feature_extractor(x)
        x = x.view(-1, 2048)
        # project back to 2048
        x = self.fc(x)
        # [batchsize , seqsize ,2048]
        x = x.view(-1,self.seq_size, x.size(1))
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:,-1,:]
        output=self.out(lstm_out)
        #output = F.normalize(output,dim=1)
        return output
        
if __name__ =='__main__':
    device = torch.device('cuda')
    model=TRN(3,126,3,7).to(device)
    dummy=torch.randn(16,3, 3, 126,126).float().to(device)
    x=model(dummy)
    print(x.shape)