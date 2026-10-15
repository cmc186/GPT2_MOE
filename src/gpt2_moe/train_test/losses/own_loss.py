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



import torch
import torch.nn as nn

class OwnLoss(nn.Module):

    def __init__(self, moe_layers:list[nn.Module], a:float = 1.0, b:float = 0.01):

        # Initialize parent
        super().__init__()
        # Define the hyperparameters
        self.a = a
        self.b = b
        # MoE layers -> read their routing state after each forward
        self.moe_layers = moe_layers
        # Language modelling loss (-100 = prompt and padding)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x:torch.Tensor, target:torch.Tensor)->torch.Tensor:

        # Language modelling loss
        ce = self.ce(x, target)

        # Load balancing loss over the MoE layers that used the router
        aux = 0.0
        for moe in self.moe_layers:
            # Forced expert -> no routing to balance
            if moe.last_probs is None:
                continue
            n_exp = moe.last_probs.shape[-1]
            # f_i: fraction of tokens sent to expert i (no gradient)
            counts = torch.bincount(moe.last_selected.flatten(), minlength = n_exp)
            f = counts.float() / counts.sum()
            # P_i: mean router probability of expert i (gradient to the gate)
            p = moe.last_probs.mean(dim = 0)
            aux = aux + n_exp * (f * p).sum()

        return self.a * ce + self.b * aux



