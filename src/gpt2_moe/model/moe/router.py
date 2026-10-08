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

        '''  
            By setting n_experts to one during training, we force
            the router to select the expert we want
        
        '''
        # Get the first dim
        first_dim = x.shape[0]
        # Input (B*N_tokens, dk) --> (B*N_tokens, NExperts)
        x = self.linear(x)
        print(f'(router) Output of the linear:{x.shape}')

        # Get the top-k over logits
        top_k_values, top_k_indices  = torch.topk(x, k = self.k, dim = 1)
        print(f'(router) Output of the top_k:{top_k_values.shape}--({top_k_indices.shape})')
        
        # Create -inf matrix
        inf_mask = torch.full_like(x, fill_value = float('-inf'))
        print(f'(router) Aspect of the -inf mask:{inf_mask.shape}')

        # Unmasking experts belonging to the top-k
        print(f'(router) BUILDING INDEXING MECHANISM')
        # 1. Create arange for indexing (B*N_tokens, 1)
        first_index = torch.arange(0, first_dim).unsqueeze(-1)
        print(f'(router) ·Aspect of the first index:{first_index.shape}')
        # 2. Second index shape (B*Ntokens, experts) -> allowed broadcast
        print(f'(router) ·Aspect of the second index:{top_k_indices.shape}')
        # 3. Set selected experts to their values 
        masked_logits = inf_mask[first_index, top_k_indices] = top_k_values
        print(f'(router) ·Aspect of the later inf maskk:{inf_mask.shape}')

        # Compute the probabilities 
        top_k_weights = self.softmax(masked_logits)
        print(f'(router) Aspect of the normalized weights:{top_k_weights.shape}')


        return top_k_weights, top_k_indices
