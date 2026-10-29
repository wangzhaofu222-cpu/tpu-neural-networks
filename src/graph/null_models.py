import numpy as np
import torch
from src.graph.stats import canonical_edges,undirected_pairs,degrees

def degree_preserving_null(edge_index,Q=10,seed=1,num_nodes=None):
    a=canonical_edges(edge_index);n=num_nodes if num_nodes is not None else int(a.max())+1
    pairs=[tuple(v) for v in undirected_pairs(a).tolist()];present=set(pairs);rng=np.random.default_rng(seed)
    attempts=int(Q*len(pairs));success=0
    if len(pairs)>=2:
        for _ in range(attempts):
            i,j=rng.choice(len(pairs),2,replace=False);u,v=pairs[i];w,z=pairs[j]
            if rng.integers(2):w,z=z,w
            if len({u,v,w,z})<4:continue
            first=tuple(sorted((u,z)));second=tuple(sorted((w,v)))
            if first in present or second in present or first==second:continue
            present.remove(pairs[i]);present.remove(pairs[j]);present.add(first);present.add(second)
            pairs[i],pairs[j]=first,second;success+=1
    out=canonical_edges(torch.tensor(pairs,dtype=torch.long).reshape(-1,2).T,n)
    exact=torch.equal(degrees(a,n),degrees(out,n));assert exact
    return out,{'Q':Q,'seed':seed,'attempts':attempts,'successful_swaps':success,
        'success_rate':success/attempts if attempts else None,'degree_sequence_exactly_preserved':exact,
        'protocol':'Q*E attempts; independent local PCG64; random second-edge orientation'}

def erdos_renyi(num_nodes,num_edges,seed):
    if not 0<=num_edges<=num_nodes*(num_nodes-1)//2:raise ValueError('Invalid edge count')
    rng=np.random.default_rng(seed);pairs=set()
    while len(pairs)<num_edges:
        u,v=sorted(rng.choice(num_nodes,2,replace=False).tolist());pairs.add((u,v))
    return canonical_edges(torch.tensor(sorted(pairs),dtype=torch.long).reshape(-1,2).T,num_nodes)
