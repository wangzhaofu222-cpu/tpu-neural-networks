import torch

def attention_statistics(a,targets):
    """Return per-example/head/query statistics, never mix heads. a: B,H,L,L."""
    l=a.shape[-1];p=torch.arange(l,device=a.device);distance=(p[:,None]-p[None,:]).abs()
    entropy=-(a*a.clamp_min(1e-30).log()).sum(-1)
    result={'entropy':entropy,'distance':(a*distance).sum(-1),
            'radius16_mass':(a*(distance<=16)).sum(-1),'max_weight':a.max(-1).values}
    valid=targets>=0
    gathered=a.gather(-1,targets.clamp_min(0)[:,None,:,None].expand(-1,a.shape[1],-1,1)).squeeze(-1)
    result['target_mass']=gathered.masked_fill(~valid[:,None,:],float('nan'))
    result['answer_entropy']=entropy.masked_fill(~valid[:,None,:],float('nan'))
    return result

def summarize(stats):
    return {name:{'per_head':torch.nanmean(v,dim=(0,2)).cpu().tolist(),
                  'per_head_query':torch.nanmean(v,dim=0).cpu().tolist()} for name,v in stats.items()}
