# Import modules
import torch
import torch.nn as nn
from .activations import GELU

class FF(nn.Module):
    def __init__(self, d_in: int, dh: int = None):
        # Initialize parent
        super().__init__()

        # Define the parameters
        dh = dh or 4 * d_in
        self.ffn = nn.Sequential(
            nn.Linear(d_in, dh),
            GELU(),
            nn.Linear(dh, d_in),
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:

        # (B, N, D_in)->(B, N, D_in)
        return self.ffn(x)
