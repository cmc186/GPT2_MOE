import torch
import torch.nn as nn 

class FF(nn.Module):
    def __init__(self, d_in: int, dh: int = None):
        # Initialize parent
        super().__init__()

        # Define the parameters
        dh = dh or 4 * d_in
        self.ffn = nn.Sequential(
            nn.Linear(d_in, dh),
            nn.GELU(),
            nn.Linear(dh, d_in),
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        
        # (B, N, d_in)->(B, N, d_in)
        output = self.ffn(x)
        print('(FFN) Computing FFN output:', output.shape)

        return output