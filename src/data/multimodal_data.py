import json,hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset,TensorDataset,DataLoader
from src.utils.io import digest,file_hash
from functools import lru_cache

@lru_cache(maxsize=2)
def _clip_identity_index(path,mtime_ns):
    manifest=json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest['status']!='PASS':raise RuntimeError('Complete genuine data audit required')
    return {c['clip_id']:{k:c[k] for k in ['clip_id','actor_id','role','source_sha256','video_frames_sha256'] if k in c} for c in manifest['clips']}

def clip_features(clip_id,root='data/lab9',manifest_path='experiments/lab9/data_manifest.json',normalization_path='experiments/lab9/feature_normalization.json',encoder_path='experiments/lab9/encoder_sources.json'):
    """One identity-checked clip call; cal is rejected before opening feature/media files."""
    from src.multimodal.cache import pack_clip_features
    mp=Path(manifest_path);c=_clip_identity_index(str(mp.resolve()),mp.stat().st_mtime_ns)[clip_id]
    if c['role']=='cal':raise ValueError('Lab9 does not read cal features/media')
    base=Path(root)/'features';v=torch.load(base/'visual'/(clip_id+'.pt'),weights_only=False);a=torch.load(base/'audio'/(clip_id+'.pt'),weights_only=False);alignment=torch.load(base/(clip_id+'_alignment.pt'),weights_only=False)
    encoders=json.loads(Path(encoder_path).read_text());stats=json.loads(Path(normalization_path).read_text());assert stats['fit_role']=='train'
    assert v['clip_id']==a['clip_id']==alignment['clip_id']==clip_id
    assert v['actor_id']==a['actor_id']==c['actor_id'] and v['role']==a['role']==c['role']
    assert v['encoder_sha256']==encoders['visual']['sha256'] and a['encoder_sha256']==encoders['audio']['sha256']
    assert v['source_sha256']==c['source_sha256'] and v['input_frames_sha256']==c['video_frames_sha256']
    result=pack_clip_features(v,a,alignment,stats);result['align'].update(actor_id=c['actor_id'],role=c['role']);return result


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
