import torch
from torch import nn

def subnet(width):
    net=nn.Sequential(nn.Linear(1,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
    nn.init.normal_(net[-1].weight,std=.01);nn.init.zeros_(net[-1].bias)
    return net

class AffineCoupling(nn.Module):
    def __init__(self,width=64,cap=4.):
        super().__init__();self.raw_s=subnet(width);self.t=subnet(width);self.cap=float(cap)
    def scale(self,a):return self.cap*torch.tanh(self.raw_s(a)/self.cap)
    def forward(self,x):
        a,b=x.split(1,-1);s=self.scale(a);y=b*torch.exp(s)+self.t(a)
        return torch.cat([a,y],-1),s.squeeze(-1)
    def inverse(self,y):
        a,b=y.split(1,-1);return torch.cat([a,(b-self.t(a))*torch.exp(-self.scale(a))],-1)

class CoordinateSwap(nn.Module):
    def forward(self,x):return x.flip(-1),x.new_zeros(x.shape[:-1])
    def inverse(self,x):return x.flip(-1)

class NormalizingFlow(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.layers=nn.ModuleList()
        for i in range(cfg['depth']):
            self.layers.append(AffineCoupling(cfg.get('width',64),cfg['cap']))
            if i<cfg['depth']-1:self.layers.append(CoordinateSwap())
    def forward(self,x):
        ld=x.new_zeros(x.shape[:-1])
        for layer in self.layers:x,term=layer(x);ld=ld+term
        return x,ld
    def inverse(self,z):
        for layer in reversed(self.layers):z=layer.inverse(z)
        return z
    def log_prob(self,x):
        z,ld=self(x);base=torch.distributions.Normal(z.new_tensor(0.),z.new_tensor(1.))
        return base.log_prob(z).sum(-1)+ld
    def sample(self,n,generator=None):
        p=next(self.parameters());z=torch.randn(n,2,dtype=p.dtype,generator=generator).to(p.device)
        return self.inverse(z)
