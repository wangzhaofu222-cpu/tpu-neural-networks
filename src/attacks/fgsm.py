import torch
from torch import nn
import torch.nn.functional as F


class NormalizedModel(nn.Module):
    def __init__(self,model,mean,std):
        super().__init__();self.model=model
        self.register_buffer('mean',torch.tensor(mean).reshape(1,3,1,1));self.register_buffer('std',torch.tensor(std).reshape(1,3,1,1))
    def forward(self,x):return self.model((x-self.mean)/self.std)


def constraint(x,adv,epsilon):
    norms=(adv-x).flatten(1).abs().max(1).values
    row={'linf_max':float(norms.max()),'pixel_min':float(adv.min()),'pixel_max':float(adv.max())}
    if row['linf_max']>epsilon+1e-6 or row['pixel_min']<0 or row['pixel_max']>1:raise RuntimeError('Attack constraint violated')
    return row


def fgsm(model,x,y,epsilon):
    if epsilon<0:raise ValueError('epsilon must be nonnegative')
    model.eval();v=x.detach().clone().requires_grad_()
    if epsilon:
        with torch.enable_grad():grad=torch.autograd.grad(F.cross_entropy(model(v),y),v)[0]
        adv=(x+epsilon*grad.sign()).clamp(0,1).detach()
    else:adv=x.detach().clone()
    return adv,{'constraints':[constraint(x,adv,epsilon)],'norms':(adv-x).flatten(1).abs().max(1).values,'epsilon':epsilon,'K':1,'R':1}
