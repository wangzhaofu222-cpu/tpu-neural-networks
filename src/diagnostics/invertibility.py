import torch

def inverse_error(flow,x):
    z,_=flow(x);back=flow.inverse(z)
    return (x-back).norm(dim=-1)/(x.norm(dim=-1)+1e-8)

def autograd_logdet(transform,x):
    x=x.detach().clone().requires_grad_(True);y,analytic=transform(x)
    rows=[torch.autograd.grad(y[:,i].sum(),x,retain_graph=True)[0] for i in range(2)]
    jac=torch.stack(rows,dim=1);sign,ld=torch.linalg.slogdet(jac)
    return {'analytic':analytic.detach(),'autograd':ld.detach(),'sign':sign.detach(),'jacobian':jac.detach(),
            'max_error':float((analytic-ld).abs().max())}
