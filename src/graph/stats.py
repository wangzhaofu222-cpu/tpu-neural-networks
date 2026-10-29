import hashlib
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def canonical_edges(edge_index,num_nodes=None):
    a=edge_index.detach().cpu().long().numpy()
    if a.size==0:return torch.empty((2,0),dtype=torch.long)
    if a.ndim!=2 or a.shape[0]!=2 or a.min()<0:raise ValueError('Invalid edge index')
    if num_nodes is not None and a.max()>=num_nodes:raise ValueError('Node out of range')
    pairs=np.sort(a,axis=0).T;pairs=np.unique(pairs[pairs[:,0]!=pairs[:,1]],axis=0)
    both=np.concatenate([pairs,pairs[:,::-1]],axis=0);both=both[np.lexsort((both[:,1],both[:,0]))]
    return torch.from_numpy(both.copy().T).long()

def undirected_pairs(edge_index):
    a=edge_index.detach().cpu().long().T
    return a[a[:,0]<a[:,1]].contiguous()

def assert_simple(edge_index,num_nodes):
    expected=canonical_edges(edge_index,num_nodes)
    if not torch.equal(edge_index.detach().cpu(),expected):raise ValueError('Expected canonical bidirectional nonloop simple graph')
    return True

def graph_hash(edge_index,num_nodes):
    a=canonical_edges(edge_index,num_nodes)
    return hashlib.sha256(str(num_nodes).encode()+a.numpy().tobytes()).hexdigest()

def degrees(edge_index,num_nodes):
    return torch.bincount(edge_index.detach().cpu()[0],minlength=num_nodes)

def graph_stats(edge_index,num_nodes,labels=None):
    assert_simple(edge_index,num_nodes);a=edge_index.cpu();d=degrees(a,num_nodes);e=a.shape[1]//2
    components=int(connected_components(coo_matrix((np.ones(a.shape[1]),(a[0].numpy(),a[1].numpy())),shape=(num_nodes,num_nodes)),directed=False,return_labels=False))
    result={'nodes':num_nodes,'undirected_edges':e,'directed_records':2*e,'mean_degree':d.double().mean().item(),
        'median_degree':float(np.median(d.numpy())),'max_degree':d.max().item(),'components':components,
        'isolated_nodes':int((d==0).sum()),'degree_sequence':d.tolist(),'graph_sha256':graph_hash(a,num_nodes)}
    if labels is not None:
        y=torch.as_tensor(labels).cpu().long();counts=torch.bincount(y);vol=torch.bincount(y,weights=d.double(),minlength=len(counts))
        pairs=undirected_pairs(a)
        result.update(class_counts=counts.tolist(),class_degree_volumes=vol.tolist(),
            hE=float((y[pairs[:,0]]==y[pairs[:,1]]).double().mean()) if e else None,
            h0=float(((counts.double()/num_nodes)**2).sum()),hdeg=float(((vol/(2*e))**2).sum()) if e else None,
            label_scope='all labels for descriptive statistics only; never used by model/hyperparameter/graph selection')
    return result
