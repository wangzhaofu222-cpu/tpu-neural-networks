import hashlib
import torch
from torch.utils.data import TensorDataset,DataLoader
from src.generative.distributions import make_roles
from src.utils.io import digest

def datasets(cfg):
    roles=make_roles(cfg['data']['target'])
    if cfg['data'].get('test_train_size') is not None:
        if cfg['experiment']['purpose']!='test':raise ValueError('Reduced generative dataset is unit-test-only')
        roles['train']=roles['train'][:cfg['data']['test_train_size']]
    hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in roles.items()}
    ds={k:TensorDataset(v,torch.zeros(len(v),dtype=torch.long)) for k,v in roles.items() if k!='test'}
    info={'dataset':cfg['data']['target'],'data_fingerprint':digest(hashes),'role_fingerprints':hashes,
        'split_fingerprint':digest({'train':[20000,601],'val':[5000,602],'test':[5000,603],'cal':'not created'}),
        'split_sizes':{'train':len(roles['train']),'val':5000,'test':5000,'cal':0},'synthetic_fixture':bool(cfg['data'].get('test_train_size')),
        'normalization':'none; original physical coordinates','domain':[[-15,15],[-15,15]],
        'data_role_protocol':'train optimization only; val NLL Flow depth choice; test excluded from training loaders'}
    return ds,info

def loaders(cfg,ds,generator):
    val_g=torch.Generator().manual_seed(cfg['experiment']['seed']+1)
    return {'train':DataLoader(ds['train'],batch_size=cfg['train']['batch_size'],shuffle=True,drop_last=True,generator=generator,num_workers=0),
            'val':DataLoader(ds['val'],batch_size=1024,shuffle=False,generator=val_g,num_workers=0)}
