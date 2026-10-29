"""Unclipped DDPM/DDIM; NFE is counted, not inferred from output shape."""
import time
import torch


def make_tau(T, K):
    if not 1 <= K <= T:
        raise ValueError('Require 1<=K<=T')
    tau = torch.unique_consecutive(torch.round(torch.arange(1, K+1, dtype=torch.float64)*T/K).long())
    if len(tau) != K or tau[0] < 1 or tau[-1] != T or not bool((tau[1:] > tau[:-1]).all()):
        raise ValueError('Invalid or duplicate tau')
    return tau.tolist()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def _sample(eps_fn, schedule, x_T, sampler, K, eta=0., z=None, keep=None, generator=None, variance='posterior', custom_tau=None):
    sync(x_T.device)
    start = time.perf_counter()
    schedule = schedule.to(x_T.device)
    a = schedule.alpha_bar.to(x_T.dtype)
    alpha = schedule.alpha.to(x_T.dtype)
    beta = schedule.beta.to(x_T.dtype)
    pv = schedule.posterior_var.to(x_T.dtype)
    tau = list(range(1, schedule.T+1)) if sampler == 'ddpm' else (list(custom_tau) if custom_tau is not None else make_tau(schedule.T, K))
    if len(tau)!=K or tau[-1]!=schedule.T or tau[0]<1 or any(type(t) is not int or t>schedule.T for t in tau) or any(a>=b for a,b in zip(tau,tau[1:])):
        raise ValueError('Invalid tau')
    keep = set(keep or [])
    saved = {schedule.T: x_T.detach().cpu().clone()} if schedule.T in keep else {}
    x = x_T.clone()
    nfe = 0
    for i in range(len(tau)-1, -1, -1):
        t = tau[i]
        previous = tau[i-1] if i else 0
        eps = eps_fn(x, torch.tensor(t, device=x.device))
        nfe += 1
        noise = None
        stochastic = previous > 0 and (sampler == 'ddpm' or eta > 0)
        if stochastic:
            noise = z[t] if z is not None else torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        if sampler == 'ddpm':
            x = (x-beta[t]/(1-a[t]).sqrt()*eps)/alpha[t].sqrt()
            if stochastic:
                x = x+(pv[t] if variance == 'posterior' else beta[t]).sqrt()*noise
        else:
            x0 = (x-(1-a[t]).sqrt()*eps)/a[t].sqrt()
            sigma = eta*((1-a[previous])/(1-a[t])*(1-a[t]/a[previous])).clamp(min=0).sqrt()
            x = a[previous].sqrt()*x0+(1-a[previous]-sigma.square()).clamp(min=0).sqrt()*eps
            if stochastic:
                x = x+sigma*noise
        if previous in keep:
            saved[previous] = x.detach().cpu().clone()
    sync(x.device)
    return x, {'NFE': nfe, 'total_predictor_calls': nfe, 'seconds': time.perf_counter()-start,
               'device': str(x.device), 'samples': len(x), 'tau': tau,
               'eta': eta if sampler == 'ddim' else None, 'sampler': sampler,
               'variance': variance if sampler == 'ddpm' else 'equation7.17',
               'clipping': False, 'saved': saved}


def ddpm_sample(eps_fn, schedule, x_T, z=None, keep=None, generator=None, variance='posterior'):
    if variance not in ['posterior', 'beta']:
        raise ValueError('Unknown variance')
    return _sample(eps_fn, schedule, x_T, 'ddpm', schedule.T, z=z, keep=keep, generator=generator, variance=variance)


def ddim_sample(eps_fn, schedule, x_T, K, eta=0., keep=None, tau=None, z=None, generator=None):
    if not 0 <= eta <= 1:
        raise ValueError('DDIM eta outside[0,1]')
    return _sample(eps_fn, schedule, x_T, 'ddim', K, eta, z, keep, generator, custom_tau=tau)
