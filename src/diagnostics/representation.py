import torch

def representation_statistics(features,normalized=False):
    x=features.detach().double();n,d=x.shape
    if n<2:raise ValueError('Need at least2 diagnostic images')
    mean=x.mean(0);centered=x-mean;cov=centered.T@centered/(n-1)
    values=torch.linalg.eigvalsh(cov).clamp_min(0).flip(0);trace=values.sum().item()
    mean_sq=x.square().sum(1).mean().item();threshold=1e-6*d*mean_sq
    collapse=trace<=threshold
    rank=None
    if not collapse:
        q=values/trace;valid=q>0;rank=float(torch.exp(-(q[valid]*q[valid].log()).sum()))
    result={'n':n,'dimension':d,'mean_norm':x.norm(dim=1).mean().item(),'mean_squared_norm':mean_sq,
            'trace_cov':trace,'eigenvalues':values.cpu().tolist(),'effective_rank':rank,
            'effective_rank_ratio':None if rank is None else rank/d,'collapse_scale':collapse,
            'collapse_threshold':threshold,'dimensional_collapse':not collapse and rank/d<.2,
            'mean_vector_norm':mean.norm().item()}
    if normalized:
        expected=n/(n-1)*(1-mean.square().sum().item())
        result.update(identity_4_9_rhs=expected,identity_4_9_error=abs(trace-expected),complete_collapse=mean.norm().item()>=.9)
    return result
