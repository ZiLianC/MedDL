import torch.nn as nn

from monai.networks.blocks.mlp import MLPBlock

class Pooling(nn.Module):


    def __init__(self, pool_size=3): 
        super().__init__() 
        self.pool = nn.AvgPool2d( pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
        
    def forward(self, x): 
        # Subtraction of the input itself is added # since the block already has a residual connection. 
        return self.pool(x) - x

class TransformerBlock(nn.Module):

    def __init__(self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Pooling()# !! Essence of POOLFORMER--POOLING
        self.norm2 = nn.LayerNorm(hidden_size)


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x