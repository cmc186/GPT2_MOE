# Basic Imports
import torch
import torch.nn as nn



# Relative imports 
from ..activations import GELU

class Expert(nn.Module):

    def __init__(self, dk:int, dh:int,  dout:int):

        # Initialize parent
        super().__init__()

        # Define the FFN
        self.l1 = nn.Linear(in_features = dk, out_features = dh)
        self.l2 = nn.Linear(in_features = dh , out_features =  dout)

        # Define the activation function
        self.activation = GELU()


    def forward(self, x:torch.Tensor)->torch.Tensor:

        # First Linear Layer (B*N, dk)-->(B*N, dhidden)
        x = self.activation(self.l1(x))
        print(f'(expert) Output of the first linear:{x.shape}')


        # Second Linear Layer (B* N, dk)-->(B*N, dhidden)
        x = self.activation(self.l2(x))
        print(f'(expert) Output of the second linear:{x.shape}')

        return x
