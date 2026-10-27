# Basic Imports 
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in:int, d_out:int, context_length:int, attr:dict):

        # Initialiaze the parent
        super().__init__() 

        # Get att specific params
        num_heads:int = attr.get('num_heads', 6)
        dropout:float = attr.get('dropout', 0.1)
        qkv_bias:bool = attr.get('qkv_bias', False)

        # Get basic parameters
        self.d_out = d_out
        self.num_heads = num_heads
        # Force sum(dim(heads)) =  dout
        self.head_dim = d_out // num_heads 

        # Define the QKV matrices -> dk = dv = dout
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal = 1)   
        )

        self.att_weights = None

    def forward(self, x):
        # Get the shape (B, N, d_in)
        b, num_tokens, d_in = x.shape
        # Define the KQV matrices (B, num_tokens, d_out)
        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We modify shapes to consider all the heads
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention 
        attn_scores = queries @ keys.transpose(2, 3)  

        # Original mask truncated to the number of tokens and converted to boolean
        # Addapted to the real amount of tokens in case input is not padded to obtain fixed N (context_length)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores -> upper diagonal MLM
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # Get attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        self.att_weights = attn_weights.detach().cpu()


        # Shape: (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where d_out -> num_heads * head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec