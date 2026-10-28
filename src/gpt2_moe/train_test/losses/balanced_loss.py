import torch
import torch.nn as nn

class BalancedLoss(nn.Module):

    def __init__(self, source:callable, params:dict):

        # Initialize parent
        super().__init__()
        # Define the hyperparameters
        self.alpha = params.get('alpha', 0.01)
        self.moe_blocks = source
        self.n_experts:int = None
        # Recieves (BxN,|V|)
        self.ce = nn.CrossEntropyLoss(ignore_index = -100)
        # Debug print
        print(f'(loss-init) Alpha Values:', self.alpha)
        print(f'(loss-init) Moe Blocks:', type(self.moe_blocks))

    def load_balancing_loss(self, probs:torch.Tensor, selected:torch.Tensor)->torch.Tensor:

        """
            Function aimed to conduct the load balancing loss
            given the following formula:

            L = N_experts*(f@p)

            Params:
                probs(torch.Tensor):
                    The probabilities tensor (tokens, experts)
                selected(torch.Tensor):
                    The selected experts per token (tokens, top_k)

            Returns:
                loss_val(torch.Tensor):
                    The balancing loss value
        
        """
        # Get amount of tokens and k size
        tokens = probs.shape[0]
        k_size = selected.shape[-1]


        #  BUILDING F (1, N)
        # 1. Create bin mask (Tokens, N)
        bin_mask = torch.zeros(size = (tokens, self.n_experts), device = probs.device)
        # 2. Populate the mask 
        bin_mask[torch.arange(0, tokens, device = selected.device).unsqueeze(1), selected] = 1
        #print(f'(lb-loss) Aspect of the populated mask:', bin_mask.shape)
        # 3. Apply sum (1, N)
        f = torch.sum(bin_mask, axis = 0, keepdim = True)
        #print(f'(lb-loss) Aspect of the raw f:', f.shape)
        # 4. Normalize by total assignations tokens*top_k_size--> row stochastic (1, N)
        f_norm = (f/(tokens*k_size))
        #print(f'(lb-loss) Aspect of the f (row stochastic):', f_norm.shape, torch.sum(f_norm))

        # BUILDING p (1, N)
        # 1. Sum probs per col (1, N)
        p_raw = torch.sum(probs, axis = 0, keepdim = True)
        #print(f'(lb-loss) Aspect of the p (raw):', p_raw.shape)
        # 2. Normalize the values (1, N)
        p_norm = (p_raw/tokens)
        #print(f'(lb-loss) Aspect of the p (normalized):', p_norm.shape, torch.sum(p_norm))

        # FINAL VALUE
        output_mat = torch.matmul(f_norm, torch.transpose(p_norm, 0, 1))
        #print(f'(lb-loss) Aspect of the output mat:', output_mat.shape, output_mat)
        loss_val = self.n_experts * (output_mat)
        return loss_val



    def filtering(self, source:torch.Tensor, target:torch.Tensor, token_id:int = -100 )->torch.Tensor:
        """  
            Function aimed to filter a token_id
            from the responses

            Params:
                source(torch.Tensor):
                    The  source tensor to build the mask
                target(torch.Tensor):
                    The target tensor to filter
                token_id(int):
                    The token id 
                    Default value is -100

            Returns:
                mod_tensor(torch.Tensor):
                    The modified Tensor
        
        """
        # Create mask  (BT, 1)
        #print('BUILDING THE MASK')
        mask = (source!= -100).squeeze()
        return target[mask] 


    def forward(self, x:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        # Compute the ce
        ce_loss = self.ce(x.flatten(0, 1), target.flatten())
        #print(f'(loss-forward) CE:', ce_loss)

        # Extract data to compute load balancing
        bal_loss_list = []
        # Check moe block length
        for i, moe in enumerate(self.moe_blocks()):
            #print(f'\nASPECT OF THE EXPERT({i})')
            # Extract the statistics and apply masking for -100
            last_probs = self.filtering(target.flatten(), moe.last_probs)
            last_selected = self.filtering(target.flatten(), moe.last_selected)
            #print(f'(loss-forward) Last probs:', last_probs.shape )
            #print(f'(loss-forward) Last selected:', last_selected.shape)
            # Populate nexperts
            if self.n_experts is None:
                self.n_experts = moe.n_exp
                #print(f'(loss-forward) Experts selected:', self.n_experts)
            # Call the loss
            bal_loss = self.load_balancing_loss(last_probs, last_selected)
            # Populate the list 
            bal_loss_list.append(bal_loss)
        
        if bal_loss_list:
            # No len in blocks
            final_bal_loss =  torch.stack(bal_loss_list).sum() * self.alpha
        else:
            final_bal_loss = torch.zeros_like(ce_loss)

        # Compute mixed loss
        mixed_loss = ce_loss + final_bal_loss  
        #print('\nFINAL REVIEW')
        #print(f'(loss-forward) Final Value CE:', ce_loss)
        #print(f'(loss-forward) Final Value BL:', final_bal_loss)
        #print(f'(loss-forward) Final Value Total:', mixed_loss)

        return mixed_loss
    
