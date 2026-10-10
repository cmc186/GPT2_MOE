# Basic Imports
import torch
import torch.nn as nn 

# File system imports
from .ffn import FF
from .moe import MOE
from .norm import LayerNorm
from .att_block import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, attr:dict, use_moe:bool = True):
        # Initialize parent
        super().__init__()
        # Common params
        self.d_in = attr["emb_dim"]
        self.d_out = attr["emb_dim"]
        self.cl = attr["context_length"]
        # Block specific params
        self.att_related = attr["attention_related"]
        self.moe_related = attr["moe_related"]
        self.use_moe = use_moe
        # Define the attention module -> ok
        self.att = MultiHeadAttention(
           self.d_in,
           self.d_out,
           self.cl,
           self.att_related)

        # Define the ff layer
        if self.use_moe:
            self.ff = MOE(
                    self.d_in, 
                    self.d_out, 
                    self.moe_related) 
        else:
            self.ff = FF(self.d_in, 
                        attr.get('ffn_hidden'))
            

        # Define normalizations (within and previous)
        self.norm1 = LayerNorm(self.d_in)
        self.norm2 = LayerNorm(self.d_in)
        
        # Define the dropout
        self.drop_shortcut = nn.Dropout(attr['dropout'])

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # Info message
        print('\nENTERING A TRANSFORMER BLOCK')
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        # (B, N, de) ->(B, N, de)
        x = self.att(x)  
        print(f'(TFBLOCK) Aspect of attention output ({x.shape})')
        x = self.drop_shortcut(x)
        # Add the original input back
        x = x + shortcut  

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        # Go through ff Layer
        x = self.ff(x)
        print(f'(TFBLOCK) Aspect of FF output ({x.shape})')
        x = self.drop_shortcut(x)
        # Add the original input back
        x = x + shortcut 

        print(f'(TFBLOCK) Aspect of block output ({x.shape})')
 

        return x