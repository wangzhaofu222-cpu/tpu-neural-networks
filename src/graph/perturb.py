import math
import numpy as np
import torch
from src.graph.stats import canonical_edges,undirected_pairs

def delete_edges(edge_index,p,seed):
    if not 0<=p<=1:raise ValueError('p outside [0,1]')
    pairs=undirected_pairs(canonical_edges(edge_index));keep=np.random.default_rng(seed).random(len(pairs))>=p
    return canonical_edges(pairs[torch.from_numpy(keep)].T)

def add_edges(edge_index,p,seed,num_nodes=None):
    if p<0:raise ValueError('Negative addition fraction')
    a=canonical_edges(edge_index);n=num_nodes if num_nodes is not None else int(a.max())+1
    pairs={tuple(v) for v in undirected_pairs(a).tolist()};target=math.floor(p*len(pairs));rng=np.random.default_rng(seed)
    if target>n*(n-1)//2-len(pairs):raise ValueError('Too many requested edges')
    added=0
    while added<target:
        i,j=sorted(rng.choice(n,2,replace=False).tolist())
        if (i,j) not in pairs:pairs.add((i,j));added+=1
    return canonical_edges(torch.tensor(sorted(pairs),dtype=torch.long).reshape(-1,2).T,n)
