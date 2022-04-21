from typing import Tuple, Union
import torch
import torch.nn as nn



class TEMPLATE(nn.Module):
    """
    xxx based on: ""
    """

    def __init__() -> None:
        """
        Args:

        Examples::

        """

        super().__init__()

    def SUB_FUNCS(self, ):
        return x

    def forward(self, x_in):
        return logits


if __name__ =='__main__':

    device = torch.device('cuda')
    model=TEMPLATE(1,2,(96,96,96)).to(device)
    dummy=torch.randn(1, 1, 96, 96,96).float().to(device)
    up=model(dummy)
    for index ,(name, param) in enumerate(model.named_parameters()):
        print( str(index) + " " +name)

