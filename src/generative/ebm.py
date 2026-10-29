import math
import torch
from torch import nn

class EnergyModel(nn.Module):
    def __init__(self,cfg):
        super().__init__();w=cfg.get('width',128);self.s=3.
        self.residual=nn.Sequential(nn.Linear(2,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,w),nn.ReLU(),nn.Linear(w,1),nn.Tanh())

    def forward(self,x):return x.square().sum(-1)/(2*self.s**2)+self.residual(x).squeeze(-1)

    def log_prob(self,x):raise NotImplementedError('EBM has no exact log_prob; use an explicitly Omega-normalized grid estimate')

    def sample(self,n,eta,temperature=1.,steps=50,boundary=((-15.,15.),(-15.,15.)),generator=None):
        from src.generative.samplers import langevin_sample
        p=next(self.parameters());x=torch.randn(n,2,generator=generator,dtype=p.dtype).to(p.device)
        return langevin_sample(self,x,eta,temperature,steps,boundary,generator)[0]

def energy_loss(energy_fn,x_pos,x_neg):
    return energy_fn(x_pos).mean()-energy_fn(x_neg.detach()).mean()

def tail_bound(R=14.,s=3.):return math.exp(1)*2*math.pi*s*s*math.exp(-R*R/(2*s*s))
