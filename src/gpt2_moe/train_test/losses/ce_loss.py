import torch
import torch.nn as nn


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, args:dict = None):
        # Initialize parent
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        # [B, T, V] -> [B*T, V]
        x = x.reshape(-1, x.shape[-1])
        target = target.reshape(-1)
        return self.ce(x, target)