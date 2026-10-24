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
        # Clamp the log  values -> avoid zeroes in log -> avoid modifiying original dist
        log_p = torch.log(torch.clamp(input = p, min = eps))
        log_q = torch.log(torch.clamp(input = q, min = eps))

        # Compute the KL over the last dim
        return (p*(log_p - log_q)).sum(-1)

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
                jsd (torch.Tensor):
                    Value for the JSD (last dim reduced)
        """
        # Get the mixture distribution M
        M = 0.5*(p + q)

        # KL(P||M) and KL(Q||M)
        kl_p = JensenDivergence.kl_divergence(p = p, q = M)
        kl_q = JensenDivergence.kl_divergence(p = q, q = M)

        # Get the final value
        return 0.5*kl_p + 0.5*kl_q
