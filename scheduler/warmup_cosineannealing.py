import math
import torch

# T_max = epoch
def warmup_CosineAnnealing(optimizer,warm_up_iter,lr_max=1e-4,lr_min=0,T_max=50):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1)
