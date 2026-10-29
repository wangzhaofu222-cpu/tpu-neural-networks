import math
import numpy as np
import torch
from scipy.stats import ncx2

class GaussianMixture2D:
    domain=((-15.,15.),(-15.,15.))
    def __init__(self,name,seed=0):
        registry={
            'two_blobs':([[-2.,0.],[2.,0.]],.3),
            'eight_gaussians':([[3*math.cos(2*math.pi*m/8),3*math.sin(2*math.pi*m/8)] for m in range(8)],.2),
            'four_corners':([[-2.5,-2.5],[-2.5,2.5],[2.5,-2.5],[2.5,2.5]],.3),
            'three_blobs':([[0.,2.5],[-2.2,-1.3],[2.2,-1.3]],.25)}
        if name not in registry:raise ValueError(name)
        centers,self.sigma=registry[name];self.name=name;self.centers=torch.tensor(centers,dtype=torch.float64)
        self.generator=torch.Generator().manual_seed(seed)

    def sample(self,n,generator=None):
        g=self.generator if generator is None else generator
        modes=torch.randint(len(self.centers),(n,),generator=g)
        return self.centers.float()[modes]+self.sigma*torch.randn(n,2,generator=g)

    def log_prob(self,x):
        x=torch.as_tensor(x)
        if not x.is_floating_point():x=x.double()
        c=self.centers.to(x);s=self.sigma
        logits=-((x[...,None,:]-c)**2).sum(-1)/(2*s*s)-math.log(2*math.pi*s*s)
        return torch.logsumexp(logits,dim=-1)-math.log(len(c))

    def mode_id(self,x):
        x=torch.as_tensor(x);dist=((x[...,None,:]-self.centers.to(x))**2).sum(-1)
        value,index=dist.min(-1);return torch.where(value<=(3*self.sigma)**2,index,-torch.ones_like(index))

    def mode_probabilities(self):
        # Disjoint radius3sigma disks: exact mixture probability via noncentral chi² CDF.
        d2=((self.centers[:,None,:]-self.centers[None,:,:])**2).sum(-1).numpy()/self.sigma**2
        mass=ncx2.cdf(9.,df=2,nc=d2).mean(1)
        return torch.tensor(np.r_[1-mass.sum(),mass],dtype=torch.float64)

def distribution(name,seed=0):return GaussianMixture2D(name,seed)

def make_roles(name):
    return {role:distribution(name,seed).sample(n) for role,n,seed in [('train',20000,601),('val',5000,602),('test',5000,603)]}
