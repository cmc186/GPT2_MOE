import torch
import torch.nn as nn

class JensenDivergence():    

    @staticmethod
    def kl_divergence(p:torch.Tensor, q:torch.Tensor, eps:float = 1e-12)->torch.Tensor:
        """
            Implementation of the KL divergence

            KL(P||Q) = sum_x(P(x)*[logP(x) - logQ(x)])

            Params:
                p (torch.Tensor):
                    The predicted distribution

                q (torch.Tensor):
                    The target distribution 

                eps (float):
                    Avoid ln(0) 

            Returns:
                kl (p|q) (torch.Tensor):
                    The output tensor
        """

        # We state two probability distributions (P and Q)
        # P: the empiric distribution (N,|V|)
        print(f'    (kl) Aspect of the empiric distribution:{p.shape}')

        # Q: the ground truth distribution: peaked in 1 token (N,|V|)
        print(f'    (kl) Aspect of the gt distribution:{q.shape}')

        # Clamp the log  values -> avoid zeroes in log -> avoid modifiying original dist
        eps = 1e-12
        log_p = torch.log(torch.clamp(input = p, min = eps))
        log_q = torch.log(torch.clamp(input = q, min = eps))

        # Compute the KL
        # Expected shape (B, N) -> zero safety was previously ensured
        kl = (p*(log_p - log_q)).sum(-1)
        print(f'    (kl) Aspect of the KL distribution:{kl.shape}')
        return  kl

    @staticmethod
    def __call__(p:torch.Tensor, q:torch.Tensor)->torch.Tensor:
        """
            Function aimed to compute the Jensen
            Divergence. The formula applied is the 
            following:

            JSD(P||Q) = 0.5*D(P||M) + 0.5*D(Q||M)

            where M is:M = (P + Q)*0.5

            Params:
                p(torch.Tensor):
                    The predicted distribution
                q(torch.Tensor):
                    The ground truth distribution
            
            Returns:
                jsd (float):
                    Value for the JSD
        """
        # Get the mixture distribution M (B, N, |V|)
        M = 0.5*(p + q)
        print(f'(jsd) Aspect of the half distribution:', M.shape)

        # KL(P||M) -->(B, N, 1)
        print(f'(jsd) GETTING KL(P||M)')
        kl_p = JensenDivergence.kl_divergence(p = p, q = M)

        # KL(Q||M) --> (B, N, 1)
        print(f'(jsd) GETTING KL(Q||M)')
        kl_q = JensenDivergence.kl_divergence(p = q, q = M)


        # Get the final value  --> (B, N, 1)  
        jsd = 0.5*kl_p + 0.5*kl_q
        print(f'(jsd) Aspect of the jensen divergence:', jsd.shape)
        return jsd