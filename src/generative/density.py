import math
import torch
from src.generative.ebm import tail_bound

def grid(dtype=torch.float64):
    axis=torch.arange(120,dtype=dtype)*.25-15+.125
    x,y=torch.meshgrid(axis,axis,indexing='xy');return torch.stack([x.reshape(-1),y.reshape(-1)],-1)

@torch.no_grad()
def density_grid(model,target,kind):
    points=grid();p=next(model.parameters());x=points.to(p);true_log=target.log_prob(points)
    if kind=='ebm':
        energy=torch.cat([model(v).detach().cpu().double() for v in x.split(2048)])
        logZ=torch.logsumexp(-energy,0)+math.log(.0625);logp=-energy-logZ
        result={'normalized_on_omega':True,'density_kind':'grid_estimate_not_exact_likelihood','log_Z_hat_omega':float(logZ),
            'Z_hat_omega':float(logZ.exp()),'tail_unnormalized_integral_upper_bound_R14':tail_bound(),
            'tail_fraction_bound_numerator_over_estimated_Z':tail_bound()/float(logZ.exp()),
            'tail_probability_rigorous_upper_bound':math.exp(2-14**2/(2*3**2)),
            'tail_note':'Eq6.3 is an unnormalized integral bound; division by gridZ uses an estimated denominator, not an exact normalization.'}
    else:
        logp=torch.cat([model.log_prob(v).detach().cpu().double() for v in x.split(2048)])
        energy=None;result={'normalized_on_omega':False,'density_kind':'exact_Flow_density_no_grid_renormalization'}
    density=logp.exp();result.update(domain=[[-15,15],[-15,15]],grid_shape=[120,120],spacing=.25,cell_area=.0625,
        Dgrid=float((density-true_log.exp()).abs().mean()),grid_integral=float(density.sum()*.0625))
    return result,{'points':points,'true_log_density':true_log,'model_log_density':logp,'energy':energy}
