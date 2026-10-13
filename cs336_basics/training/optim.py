from re import L
import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    # a step's job is to update all the weight tensors, and I think torch handles calculating the loss for each param in the closure()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # get the learning rate from the group
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # state associated with the group's params
                t = state.get("t", 0) # get the iteration number from the state, default 0 (I guess torch handles this)
                grad = p.grad.data # get the gradient of the loss, again I guess torch sets .grad.data based on the backwards pass gradients
                p.data -= lr / math.sqrt(t + 1) * grad # update the weight tensor in place 
                state["t"] = t + 1 # ok i guess WE increment the interation number, something must define what "t" means in the group's state dict
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=[0.9,0.999], weight_decay=0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss= None if closure is None else closure()
        for group in self.param_groups:
            # get the global values, which are model hyperparameters
            lr = group["lr"]
            betas = group["betas"]
            B1 = betas[0]
            B2 = betas[1]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                # update the adjusted, per parameter tensor, hyperparameters (adjusted lr, moments, then use allat to adjust the weights)
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0) + 1
                state["t"] = t
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data
                lr_adj = lr * (math.sqrt(1-B2**t)/(1-B1**t))
                data = p.data
                data = data - lr * weight_decay * data
                m = B1*m + (1-B1)*grad
                v = B2*v + (1-B2)*grad**2
                data = data - lr_adj*(m/(torch.sqrt(v) + eps))
                p.data = data
                state["m"] = m
                state["v"] = v

        return loss

# some standalone functions that we will drop in our training loop that update global parameters: learning rate and the gradients themselves.
def learning_rate_schedule(t: int, lr_max: float, lr_min: float, T_warmup: int, T_final: int) -> float:
    if t < T_warmup:
        return (t/T_warmup)*lr_max
    elif (T_warmup <= t) and (t <= T_final):
        return lr_min + (1/2)*(1 + math.cos((t - T_warmup)*math.pi/(T_final - T_warmup)))*(lr_max-lr_min)
    else:
        return lr_min

def gradient_clipping(parameters: [torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    # modify the parameters in place
    # to do this, since parameters may be given as a generator, make it into a list for the first pass
    parameters_list = list(parameters)
    sum_of_squares = 0
    for param in parameters_list:
        if param.grad is not None:
            sum_of_squares += torch.sum(torch.square(param.grad.data))
    norm = math.sqrt(sum_of_squares)
    if norm >= max_l2_norm:
        for param in parameters_list:
            if param.grad is not None:
                param.grad.data = param.grad.data * (max_l2_norm/(norm + eps))
