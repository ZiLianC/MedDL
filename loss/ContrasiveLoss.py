import torch.nn.functional as F
import torch


def contrastive_loss(pred1,pred2,target):
    MARGIN = 2
    euclidean_dis = F.pairwise_distance(pred1,pred2)
    target = target.view(-1)
    loss = (1-target)*torch.pow(euclidean_dis,2) + target * torch.pow(torch.clamp(MARGIN-euclidean_dis,min=0),2)
    return loss
