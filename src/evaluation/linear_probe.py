"""Probe training dispatches to the same ExperimentRunner, using cached fixed features."""
from pathlib import Path
import torch
from src.utils.io import digest

class FeatureRows(torch.utils.data.Dataset):
    def __init__(self,x,y):self.x,self.y=x,y
    def __len__(self):return len(self.y)
    def __getitem__(self,i):return self.x[i],self.y[i]

def datasets(cfg):
    data=torch.load(cfg['data']['feature_path'],map_location='cpu',weights_only=False)
    if data['train']['role']!='train' or data['val']['role']!='val':raise PermissionError('Probe data role violation')
    indices=cfg['data']['label_indices'];all_ids=data['train']['indices'];position={v:i for i,v in enumerate(all_ids)}
    selected=[position[i] for i in indices];kind=cfg['data']['feature_kind']
    train=data['train'][kind][selected].float();val=data['val'][kind].float()
    # Train-only feature standardization is shared by every encoder/probe/fraction.
    mean=train.mean(0);std=train.std(0).clamp_min(1e-5)
    ds={'train':FeatureRows((train-mean)/std,data['train']['labels'][selected]),'val':FeatureRows((val-mean)/std,data['val']['labels'])}
    info={'dataset':'cached frozen encoder features','data_fingerprint':digest({'cache':data['fingerprint'],'indices':indices,'kind':kind}),
          'split_fingerprint':data['split_fingerprint'],'label_indices':indices,'encoder_RUN_ID':data['encoder_RUN_ID'],
          'encoder_state_sha256':data['train']['encoder_state_sha256'],'normalization':{'feature_mean':mean.tolist(),'feature_std':std.tolist(),'source':'selected train only'},
          'actual_train_size':len(selected),'synthetic_fixture':False}
    return ds,info

def loaders(cfg,ds,generator):
    device=cfg['runtime']['device']
    # One full batch avoids repeated per-image Python collation; training still uses the existing loop.
    return {role:[(v.x.to(device),v.y.to(device))] for role,v in ds.items()}
