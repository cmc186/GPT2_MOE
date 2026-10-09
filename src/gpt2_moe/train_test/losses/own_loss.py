import torch
import torch.nn as nn

class OwnLoss(nn.Module):

    def __init__(self, a:float, b:float):

        # Initialize parent
        super().__init__()
        # Define the hyperparameters
        self.a = a
        self.b = b

    def forward(self, x:torch.Tensor, target:torch.Tensor)->float:
        return None
        
        


