import json,hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset,TensorDataset,DataLoader
from src.utils.io import digest,file_hash


class MelDataset(Dataset):
    def __init__(self,clips,stats):self.clips=clips;self.mean=torch.tensor(stats['mean'])[:,None];self.std=torch.tensor(stats['std'])[:,None]
    def __len__(self):return len(self.clips)
    def __getitem__(self,i):
        c=self.clips[i];mel=torch.load(c['log_mel_path'],weights_only=True);normalized=(mel-self.mean)/self.std
        return torch.stack([normalized,torch.ones_like(normalized)]),c['emotion']


def mel_collate(items):
    n=max(x.shape[-1] for x,y in items);x=torch.zeros(len(items),2,64,n)
    for i,(a,_) in enumerate(items):x[i,:,:,:a.shape[-1]]=a
    return x,torch.tensor([y for _,y in items])


def datasets(config):
    cfg=config['data'];manifest=json.loads(Path(cfg['manifest_path']).read_text(encoding='utf-8'))
    if manifest['status']!='PASS':raise RuntimeError('Genuine complete RAVDESS data audit required before training')
    if manifest.get('synthetic_fixture') and config['experiment']['purpose']!='test':raise ValueError('Synthetic fixtures cannot become scientific RAVDESS results')
    if cfg['name']=='ravdess_audio':
        stats=json.loads(Path(cfg['normalization_path']).read_text());clips=manifest['clips'];ds={role:MelDataset([c for c in clips if c['role']==role],stats) for role in ['train','val']}
        data_hash=digest({'manifest':file_hash(cfg['manifest_path']),'mel_stats':file_hash(cfg['normalization_path']),'raw_source':manifest['raw_identity_hash']})
    else:
        feature=torch.load(cfg['feature_path'],weights_only=False);ds={role:TensorDataset(feature[role]['x'],feature[role]['y']) for role in ['train','val']}
        data_hash=file_hash(cfg['feature_path'])
    info={'dataset':'test_fixture' if manifest.get('synthetic_fixture') else 'RAVDESS official speech, frozen feature/AudioCNN protocol','synthetic_fixture':bool(manifest.get('synthetic_fixture')),'data_fingerprint':data_hash,'split_fingerprint':manifest['split_fingerprint'],'split_sizes':{r:sum(c['role']==r for c in manifest['clips']) for r in ['train','val','cal','test']},'teacher_split_status':'TEACHER_SPLIT_UNVERIFIED','cal_read':False,'manifest_sha256':file_hash(cfg['manifest_path'])}
    return ds,info


def loaders(config,ds,generator):
    args={'batch_size':config['train']['batch_size'],'num_workers':0}
    if config['data']['name']=='ravdess_audio':args['collate_fn']=mel_collate
    return {'train':DataLoader(ds['train'],shuffle=True,generator=generator,**args),'val':DataLoader(ds['val'],shuffle=False,**args)}
