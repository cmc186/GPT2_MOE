# Basic Imports
import torch
import torch.nn as nn


# Relative imports 
from .router import Router
from .expert import Expert


class MOE(nn.Module):

    def __init__(self, args:dict[str, int]):
        # Initialize the main module 
        super().__init__()

        # Get the args
        self.input_dim = args.get('input_dim', 768)
        self.n_exp = args.get('experts', 1)
        self.k_routing = args.get('k_routing', 1)
        self.emb_dim = args.get('emb_dim', 768 )

        
        # Initialize the gating mechanism
        self.gate = Router(dk = self.input_dim, n_experts = self.n_exp, k = self.k_routing)

        # Initialize the experts
        self.experts:list = [Expert(dk = self.input_dim, dh = self.emb_dim, dout = self.input_dim) for _ in range(self.n_exp)]


    def forward(self, x:torch.Tensor)->torch.Tensor:

        # Keep original shape
        n_batches, tokens,_ = x.shape
        #1. Get the top-k experts
        print(f'(MOE) CALLING GATING MECHANISM')
        # Flatten before passing (B, N, din)-->(B*N, di)
        x =  x.view(-1, self.input_dim)
        # Results size (B*N, din)- (B*N, din)
        weights, selected_experts = self.gate(x)

        # 2. Design the results tensor (B*N, dout = din)
        results_tensor = torch.zeros_like(x)
        print(f'(MOE) Aspect of the container for output tensor:', results_tensor.shape)

        # 3. Iterate over the experts
        for i, expert in enumerate(self.experts):
            print(f'(MOE) EXPERT ({i})')

            # Get matching tokens
            token_idx, expert_idx  =  torch.where(selected_experts == i)
            # Check if it is not none -> avoid calling not used experts 
            if token_idx is None:
                continue
            # Get expert output (batch*N, dout)
            expert_output = expert(x[token_idx, :])
            # Get expert weights (N, 1)
            expert_weigths = weights[token_idx, expert_idx].unsqueeze(-1)
            print(f'(expert) Aspect of the expert weights:', expert_weigths.shape)

            # Modify the current results tracking --> (N, 1)*(N, )
            results_tensor[token_idx] += expert_weigths*expert_output

        #4. Reshape the  output (B*N, din = dout) - >(B, N, din = dout)
        final_output = results_tensor.view(n_batches, tokens, self.input_dim)
        print(f'(MOE) Aspect of the  output tensor:', final_output.shape)

