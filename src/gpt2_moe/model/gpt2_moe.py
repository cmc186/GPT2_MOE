import torch
import torch.nn as nn 


# File System Imports
from .norm import LayerNorm
from .tf_block import TransformerBlock
from .model_utils import moe_mask

class GPTMOE(nn.Module):
    def __init__(self, attr:dict):
        # Initialize parents
        super().__init__()

        # Compute the moe mask
        self.moe_mask  = moe_mask(attr.get('n_layers'), **attr.get("moe_mask"))

        # Define embedding layer
        self.tok_emb = nn.Embedding(attr["vocab_size"], attr["emb_dim"])
        self.pos_emb = nn.Embedding(attr["context_length"], attr["emb_dim"])
        self.drop_emb = nn.Dropout(attr["dropout"])

        # Build the transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(attr, use_moe = m) for m in self.moe_mask])


        # Output layer norm
        self.final_norm = LayerNorm(attr["emb_dim"])
        # Language head
        self.out_head = nn.Linear(
            attr["emb_dim"], attr["vocab_size"], bias=False
        )

    def forward(self, in_idx:torch.Tensor)->torch.Tensor:

        # Get input dims (B, N)
        batch_size, seq_len = in_idx.shape
        # Embed the inputs (B, N, de)
        # 1. Get contextual embeddings
        tok_embeds = self.tok_emb(in_idx)
        
        # 2. Get positional embeddings
        print(f'(GPT)CREATING EMBEEDINGS')
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        print(f'(GPT)Aspect of the embeddings:{x.shape}')

        # 3. Go trought transformer blocks (B, N, de) -> (B, N, de)
        x = self.trf_blocks(x)
        print(f'(GPT)Output of the trf blocks:{x.shape}')
        x = self.final_norm(x)

        # Get logits (B, N, de) ->(B, N, |V|)
        logits = self.out_head(x)
        print(f'(GPT)Aspect of the logits space:{x.shape}')

        return logits