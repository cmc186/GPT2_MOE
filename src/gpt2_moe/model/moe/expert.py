# Basic Imports
import torch
import torch.nn as nn



# Relative imports
from ..activations import GELU

class Expert(nn.Module):

    def __init__(self, dk:int, dh:int,  d_out:int):

        # Initialize parent
        super().__init__()

        # Define the FFN
        self.l1 = nn.Linear(in_features = dk, out_features = dh)
        self.l2 = nn.Linear(in_features = dh , out_features =  d_out)

        # Define the activation function
        self.activation = GELU()


    def forward(self, x:torch.Tensor)->torch.Tensor:

        # First Linear Layer (B*N, dk)-->(B*N, dhidden)
        x = self.activation(self.l1(x))


        # Second Linear Layer (B* N, dk)-->(B*N, dhidden)
        x = self.l2(x)

        return x
