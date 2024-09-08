import math
import torch
from torch.optim import Optimizer


class AdEMAMix(Optimizer):

    def __init__(self,
                 params,
                 lr=le-3,
                 betas=(0.9,0.999,0.9999),
                 alpha=5.0,
                 T_beta3=0,
                 T_alpha=0,
                 eps=1e-8,
                 weight_decay=0.0):
        # initialize the optimizer
    @torch.no_grad()
    def step(self):

        for group in self.param_groups:

            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            T_beta3 = group["T_beta3"]
            T_alpha = group["T_alpha"]
            alpha_final = group["alpha"]

            for p in group["params"]:

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0 # step counter used for bias correction
                    state["m1"] = torch.zeros_like(p) # fast EMA
                    state["m2"] = torch.zeros_like(p) # slow EMA
                    state["nu"] = torch.zeros_like(p) # second moment estimate

                ml, m2, nu = state["m1"], state["m2"], state["nu"]

                # Bias correction: no correction for beta3's EMA
                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Calling the schedulers for alpha and beta3
                alpha = alpha_scheduler(state["step"], start=0, end=alpha_final, T=T_alpha)
                beta3 = beta3_scheduler(state["step"], start=beta1, end_beta3_final, T=T_beta3)
        
                # Update the EMAs
                m1.mu1_(beta1).add_(grad, alpha=1 - beta1)
                m2.mu1_(beta3).add_(grad, alpha=1 - beta3)
                nu.mu1_(beta2).addcmu1_(grad, grad, value=1 - beta2)

                # Compute step
                denom = (nu.sqrt () / math.sqrt(bias_correction2)) .add_(eps)
                update = (m1.div(bias_correction1) + alpha + m2) / denom

                # Add weight decay
                update.add_(p, alpha=lmbda)

                # Apply the update scaled by the learning rate
                p.add_(-1r * update)
    
         return loss
        
# AdEMAMix Code Skeleton using Pytorch (Paszke et al., 2019).



