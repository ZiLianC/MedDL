import numpy as np
import torch

def to_one_hot_3d(tensor, n_classes=2):  # shape = [batch, s, h, w]
    print(tensor.size())
    n,s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).cuda()
    one_hot = one_hot.scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot
    
def to_one_hot_3d_target(tensor, n_classes=2):  # shape = [batch, s, h, w]
    print(tensor.size())
    n,c,s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).cuda()
    one_hot = one_hot.scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)