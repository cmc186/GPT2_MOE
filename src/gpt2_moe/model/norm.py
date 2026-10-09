# Basic Imports
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim:int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
            Scales data per instance and makes
            it follow a ~N(0, 1)
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift