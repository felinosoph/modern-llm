import torch
import torch.nn as nn
import torch.nn.functional as F 

#unused just for reference to understand the underlying logic!

class RMSNorm(nn.Module): 
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x*self.weight)/rms
