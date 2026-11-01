import torch
import torch.nn.functional as F
from src.attacks.fgsm import constraint


def pgd(model,x,y,epsilon,K=10,R=1,alpha=None,seed=1):
    if epsilon<0 or K<1 or R<1:raise ValueError('Invalid PGD parameters')
    model.eval();alpha=2.5*epsilon/K if alpha is None else alpha
    g=torch.Generator(device=x.device).manual_seed(seed);best=x.detach().clone();loss_best=torch.full((len(x),),-torch.inf,device=x.device);checks=[];restart_losses=[]
    for r in range(R):
        adv=(x+torch.empty_like(x).uniform_(-epsilon,epsilon,generator=g)).clamp(0,1).detach()
        checks.append({'restart':r,'step':0,**constraint(x,adv,epsilon)})
        for step in range(K):
            with torch.enable_grad():
                adv.requires_grad_();grad=torch.autograd.grad(F.cross_entropy(model(adv),y),adv)[0]
            adv=torch.minimum(torch.maximum(adv.detach()+alpha*grad.sign(),(x-epsilon).clamp(0,1)),(x+epsilon).clamp(0,1)).detach()
            checks.append({'restart':r,'step':step+1,**constraint(x,adv,epsilon)})
        with torch.no_grad():loss=F.cross_entropy(model(adv),y,reduction='none')
        restart_losses.append(loss.detach().cpu());mask=loss>loss_best;best[mask]=adv[mask];loss_best=torch.maximum(loss_best,loss)
    return best.detach(),{'constraints':checks,'norms':(best-x).flatten(1).abs().max(1).values,'epsilon':epsilon,'K':K,'R':R,'alpha':alpha,'random_start':True,'restart_losses':torch.stack(restart_losses),'selected_losses':loss_best.detach().cpu()}
