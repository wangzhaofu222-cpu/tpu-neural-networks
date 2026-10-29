import torch


@torch.no_grad()
def score_field(eps_fn,exact,schedule,points,t):
    star=exact.score(points,t)
    net=-eps_fn(points,torch.tensor(t,device=points.device))/(1-schedule.alpha_bar[t]).sqrt().to(points)
    sn=star.norm(dim=1);nn=net.norm(dim=1)
    cosine=(star*net).sum(1)/(sn*nn).clamp(min=1e-30)
    ratio=nn/sn.clamp(min=1e-30)
    weights=torch.softmax(exact.log_prob(points,t),0)
    return {'t':t,'points':points.cpu(),'exact':star.cpu(),'network':net.cpu(),
            'cosine':cosine.cpu(),'length_ratio':ratio.cpu(),'weights':weights.cpu(),
            'weighted_cosine':float((weights*cosine).sum()),'weighted_length_ratio':float((weights*ratio).sum())}
