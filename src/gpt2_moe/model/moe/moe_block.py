# Basic Imports
import torch
import torch.nn as nn


# Relative imports
from .router import Router
from .expert import Expert

class MOE(nn.Module):

    def __init__(self, d_in:int, d_out:int, args:dict[str, int]):
        # Initialize the main module
        super().__init__()

        # Get common params
        self.d_in = d_in
        self.d_out = d_out

        # Get the args
        self.n_exp = args.get('experts', 1)
        self.k_routing = args.get('k_routing', 1)
        # Working with as much params as the original 
        self.hidden_dim = args.get('hidden_dim', (4 * self.d_in)//self.k_routing) 


        # Initialize the gating mechanism
        self.gate = Router(dk = self.d_in, n_experts = self.n_exp, k = self.k_routing)

        # Initialize the experts -> modulelist to include their parameters within model params
        self.experts:nn.ModuleList = nn.ModuleList([Expert(dk = self.d_in, dh = self.hidden_dim, d_out = self.d_out) for _ in range(self.n_exp)]
                                                )

        # Forced expert index (None -> use the router)
        self.forced_expert = None

        # Last routing state -> in case we use balancing loss
        self.last_probs = None
        self.last_selected = None


    def forward(self, x:torch.Tensor)->torch.Tensor:

        # 0. Forced expert -> skip the router
        if self.forced_expert is not None:
            return self.experts[self.forced_expert](x)

        # 1. Get the top-k experts
        n_batches, tokens, _ = x.shape

        # Flatten before passing (B, N, D_in)-->(B*N, D_in)
        x = x.reshape(-1, self.d_in)
        # Results size (B*N, k), (B*N, k), (B*N, N_experts)
        weights, selected_experts, probs = self.gate(x)

        # Keep routing state for the aux loss
        self.last_probs = probs
        self.last_selected = selected_experts

        # 2. Design the results tensor (B*N, D_out = D_in)
        results_tensor = torch.zeros_like(x)

        # 3. Iterate over the experts
        for i, expert in enumerate(self.experts):
            # Get matching tokens
            token_idx, expert_idx = torch.where(selected_experts == i)
            if token_idx.numel() == 0:
                continue

            # Get expert output (tokens_i, D_out)
            expert_output = expert(x[token_idx, :])
            # Get expert weights (tokens_i, 1)
            expert_weigths = weights[token_idx, expert_idx].unsqueeze(-1)
            # Accumulate weighted output
            results_tensor[token_idx] += expert_weigths*expert_output

        # 4. Reshape the output (B*N, D_in = D_out) -> (B, N, D_in = D_out)
        final_output = results_tensor.view(n_batches, tokens, self.d_out)

        return final_output
