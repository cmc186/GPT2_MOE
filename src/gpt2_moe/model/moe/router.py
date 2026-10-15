# Basic Imports
import torch
import torch.nn as nn



class Router(nn.Module):
    def __init__(self, dk:int, n_experts:int, k:int):


        # Initialize parent
        super().__init__()

        # Define the parameters
        self.n_experts = n_experts

        self.k = min(k, self.n_experts)

        # Define the linear layer (B*N, dk) --> (B*N, nExperts)
        self.linear = nn.Linear(in_features = dk, out_features = n_experts)

        # Define the softmax
        self.softmax = nn.Softmax(dim = -1)




    def forward(self, x:torch.Tensor)->torch.Tensor:

        # Input (B*N_tokens, dk) --> (B*N_tokens, N_experts)
        logits = self.linear(x)

        # Probabilities over all the experts (B*N_tokens, N_experts)
        probs = self.softmax(logits)

        # Get the top-k over probabilities (B*N_tokens, k)
        top_k_probs, top_k_indices = torch.topk(probs, k = self.k, dim = -1)

        # Normalize within the top-k experts.
        if self.k > 1:
            top_k_weights = top_k_probs / top_k_probs.sum(dim = -1, keepdim = True)
        else:
            top_k_weights = top_k_probs

        return top_k_weights, top_k_indices, probs
