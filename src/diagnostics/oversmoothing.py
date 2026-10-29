import math
import numpy as np
import torch
from src.graph.stats import degrees,undirected_pairs
from src.diagnostics.gradients import gradient_statistics

def sample_pairs(n,count=20000,seed=1729):
    count=min(count,n*(n-1)//2);rng=np.random.default_rng(seed);pairs=set()
    while len(pairs)<count:
        u,v=sorted(rng.choice(n,2,replace=False).tolist());pairs.add((u,v))
    return torch.tensor(sorted(pairs),dtype=torch.long).reshape(-1,2)

def cosine_similarity(h,pairs):
    z=h.detach().cpu().double();z=z/z.norm(dim=1,keepdim=True).clamp_min(1e-300)
    # A pair involving a zero vector contributes zero, including two zeros.
    total=0.
    for p in pairs.split(1024):total+=(z[p[:,0]]*z[p[:,1]]).sum().item()
    return total/len(pairs)

def energy(h,edge_index):
    z=h.detach().cpu().double();scale=z.abs().max()
    if scale==0:return None
    z=z/scale;den=z.square().sum();d=degrees(edge_index,len(z)).double()+1
    z=z/d.sqrt()[:,None];p=undirected_pairs(edge_index)
    return float((z[p[:,0]]-z[p[:,1]]).square().sum()/den)

def normalized_operator(edge_index,n):
    a=edge_index.cpu();loops=torch.arange(n);idx=torch.cat([a,torch.stack([loops,loops])],1)
    d=degrees(a,n).double()+1;v=(d[idx[0]]*d[idx[1]]).rsqrt()
    return torch.sparse_coo_tensor(idx,v,(n,n)).coalesce()

def gradient_record(model):
    raw=gradient_statistics(model)
    # Lab1 definition, restricted to comparable convolution weight matrices (not biases).
    selected={k:v['per_parameter']['weight']['g'] for k,v in raw.items()
              if k.startswith('layers.') and 'per_parameter' in v and 'weight' in v['per_parameter']}
    g=list(selected.values());gi=math.log10((max(g)+1e-12)/(min(g)+1e-12)) if g else None
    return {'raw':raw,'weight_g':selected,'GI':gi,'GI_scope':'layer weight RMS; Lab1 gradient_statistics unchanged; epsilon=1e-12'}

def select_depth(rows):
    """Declared profile order breaks validation ties, without reading test metrics."""
    return max(rows,key=lambda row:row['val'])['depth']

def classify_depth(rho,gi,e,probe,best_e,best_probe):
    if rho>=.9:return 'undertrained'
    if gi is not None and gi>=6:return 'gradient_imbalance; smoothing attribution not established'
    if rho<=.5 and e>=best_e:return 'overfitting'
    if rho<=.5 and e<best_e and probe<best_probe:return 'smoothing'
    return 'mixed'
