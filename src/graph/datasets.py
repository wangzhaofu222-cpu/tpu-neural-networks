"""Full-graph transductive batches; training exposes train/val targets only."""
from pathlib import Path
import torch
from src.utils.io import file_hash,digest
from src.graph.stats import graph_stats

def load_graph(path):
    return torch.load(path,map_location='cpu',weights_only=False)

def fingerprints(g):
    import hashlib
    h=lambda t:hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()
    return {k:h(g[k]) for k in ['x','y','train_mask','val_mask','test_mask']}

def datasets(cfg):
    g=load_graph(cfg['data']['graph_path']);n=len(g['y'])
    masks=[g[k+'_mask'] for k in ['train','val','test']]
    if any(m.dtype!=torch.bool or len(m)!=n for m in masks) or (sum(m.long() for m in masks)>1).any():
        raise ValueError('Invalid/disjoint role masks')
    if cfg['data']['name']=='graph_features':
        features=torch.load(cfg['data']['features_path'],weights_only=False)
        # Standardization is fitted using train-role nodes only.
        mean=features[g['train_mask']].mean(0);std=features[g['train_mask']].std(0)
        # StandardScaler convention: a constant train column has unit scale.
        # Never amplify a feature seen only outside train by division by epsilon.
        std=torch.where(std<1e-6,torch.ones_like(std),std)
        x=(features-mean)/std
    else:x=g['x']
    ds={}
    for role in ['train','val']:
        y=torch.full_like(g['y'],-100);m=g[role+'_mask'];y[m]=g['y'][m]
        ds[role]=(x,y)
    f=fingerprints(g)
    info={'dataset':'Cora Planetoid public; transductive','synthetic_fixture':cfg['experiment']['purpose']=='test',
          'data_fingerprint':digest({'graph_file':file_hash(cfg['data']['graph_path']),
              'features_file':file_hash(cfg['data']['features_path']) if cfg['data']['name']=='graph_features' else None}),
          'split_fingerprint':digest({k:f[k] for k in ['train_mask','val_mask','test_mask']}),
          'split_sizes':{**{r:int(g[r+'_mask'].sum()) for r in ['train','val','test']},'cal':0},
          'graph_statistics':graph_stats(g['edge_index'],n,g['y']), 'immutable_node_fingerprints':f,
          'teacher_split_independently_verified':False,'split_substitution':'user authorized PyG public masks 2026-10-29',
          'feature_preprocessing':'row L1 NormalizeFeatures (no label access)',
          'role_protocol':'full X/edges visible; only train targets in objective; only val targets in selection; test excluded from loaders'}
    return ds,info

def loaders(cfg,ds,generator):
    return {role:[batch] for role,batch in ds.items()}
