"""Frozen, identity-checked feature extraction; never opens cal media."""
import argparse,json,time
from pathlib import Path
import torch
import yaml
from src.models.small_cnn import build_model
from src.utils.io import write_json,file_hash,digest
from src.utils.reproducibility import seed_all
from src.multimodal.alignment import alignment_windows
from src.multimodal.audio import log_mel

R=Path('experiments/lab9');D=Path('data/lab9');CACHE=D/'features'
def load_visual(device='cuda'):
    s=json.loads((R/'encoder_sources.json').read_text())['visual']
    assert file_hash(s['path'])==s['sha256']
    m=build_model(s['config']['model']);m.load_state_dict(torch.load(s['path'],map_location='cpu',weights_only=False)['model']);m.requires_grad_(False).eval().to(device)
    return m,s

@torch.no_grad()
def visual_features(frames,m,s,blur=0):
    from torchvision.transforms.functional import gaussian_blur
    device=next(m.parameters()).device
    mean=torch.tensor(s['normalization']['mean'],device=device)[None,:,None,None];std=torch.tensor(s['normalization']['std'],device=device)[None,:,None,None]
    z=[]
    for b in frames.split(8):
        if not len(b):continue
        x=b.to(device).float()/255
        if blur:x=gaussian_blur(x,[5,5],sigma=blur)
        z.append(m.forward_features((x-mean)/std).cpu())
    return torch.cat(z) if z else torch.zeros(0,512)

def visual_stream():
    seed_all(1,True,2);m,s=load_visual();out=CACHE/'visual';out.mkdir(parents=True,exist_ok=True);start=time.time()
    while True:
        for p in sorted((D/'preprocessed').glob('*/metadata.json')):
            c=json.loads(p.read_text());target=out/(c['clip_id']+'.pt')
            if target.exists():
                old=torch.load(target,weights_only=False)
                if old['encoder_sha256']!=s['sha256'] or old['input_frames_sha256']!=c['video_frames_sha256']:raise ValueError('Stale visual cache: encoder/frame identity changed')
                continue
            assert c['role']!='cal'
            v=torch.load(c['video_frames_path'],weights_only=True)
            z=visual_features(v['frames'],m,s)
            item={'clip_id':c['clip_id'],'actor_id':c['actor_id'],'role':c['role'],'source_sha256':c['source_sha256'],'encoder_sha256':s['sha256'],'z':z,'indices':v['video_indices_5fps'],'input_frames_sha256':c['video_frames_sha256'],'available':len(z)>0}
            temp=target.with_suffix('.tmp');torch.save(item,temp);temp.replace(target)
            if len(list(out.glob('*.pt')))%60==0:print('VISUAL',len(list(out.glob('*.pt'))),'seconds',round(time.time()-start,1),flush=True)
        if (R/'preprocessing_complete.json').exists() and len(list(out.glob('*.pt')))==1320:break
        time.sleep(3)
    write_json(R/'visual_features_complete.json',{'clips':1320,'cal_read':False,'encoder_sha256':s['sha256'],'wall_seconds':time.time()-start,'frozen':True})

def load_audio(device='cuda'):
    s=json.loads((R/'encoder_sources.json').read_text())['audio'];assert s['status']=='PASS'
    cfg=yaml.safe_load((Path(s['run_path'])/'config.resolved.yaml').read_text(encoding='utf-8'))
    m=build_model(cfg['model']);m.load_state_dict(torch.load(s['path'],map_location='cpu',weights_only=False)['model']);m.requires_grad_(False).eval().to(device)
    assert file_hash(R/'mel_normalization.json')==s['normalization_sha256'],'Audio encoder train normalization changed'
    stats=json.loads((R/'mel_normalization.json').read_text());return m,s,stats

@torch.no_grad()
def audio_features(mel,m,stats):
    mean=torch.tensor(stats['mean'])[:,None];std=torch.tensor(stats['std'])[:,None]
    x=torch.stack([(mel-mean)/std,torch.ones_like(mel)])[None].to(next(m.parameters()).device)
    z,valid=m.forward_features(x);return z[0,valid[0]].cpu()

def finish():
    seed_all(1,True,2);manifest=json.loads((R/'data_manifest.json').read_text());assert manifest['status']=='PASS'
    m,s,stats=load_audio();out=CACHE/'audio';out.mkdir(parents=True,exist_ok=True);roles={r:[] for r in ['train','val','test']}
    for c in manifest['clips']:
        if c['role']=='cal':continue
        p=out/(c['clip_id']+'.pt')
        if not p.exists():
            z=audio_features(torch.load(c['log_mel_path'],weights_only=True),m,stats)
            torch.save({'clip_id':c['clip_id'],'actor_id':c['actor_id'],'role':c['role'],'encoder_sha256':s['sha256'],'dataset_hash':manifest['raw_identity_hash'],'z':z,'normalization_sha256':file_hash(R/'mel_normalization.json')},p)
        a=torch.load(p,weights_only=False);v=torch.load(CACHE/'visual'/(c['clip_id']+'.pt'),weights_only=False)
        assert a['clip_id']==v['clip_id']==c['clip_id'] and a['encoder_sha256']==s['sha256']
        assert a['normalization_sha256']==file_hash(R/'mel_normalization.json')
        assert v['input_frames_sha256']==c['video_frames_sha256'] and v['source_sha256']==c['source_sha256']
        assert v['encoder_sha256']==json.loads((R/'encoder_sources.json').read_text())['visual']['sha256']
        aligned=alignment_windows(v['indices'],len(a['z']),av_offset_ms=c['av_offset_ms'])
        torch.save({'clip_id':c['clip_id'],'visual_original_indices':v['indices'],**aligned},CACHE/(c['clip_id']+'_alignment.pt'))
        # Independent temporal masked means (Eq9.5); alignment remains available for frame-level fusion.
        roles[c['role']].append((c,torch.cat([v['z'].mean(0) if len(v['z']) else torch.zeros(512),a['z'].mean(0)]),float(v['available'])))
    train=torch.stack([x for c,x,v in roles['train']]);mean=train.mean(0);std=train.std(0,unbiased=False).clamp_min(1e-6)
    write_json(R/'feature_normalization.json',{'mean':mean.tolist(),'std':std.tolist(),'fit_role':'train','clip_ids':[c['clip_id'] for c,x,v in roles['train']],'ddof':0})
    packed={}
    for r,rows in roles.items():
        x=torch.stack([(z-mean)/std for c,z,v in rows]);available=torch.tensor([[v,1.] for c,z,v in rows]);x[:,:512]*=available[:,:1]
        packed[r]={'x':torch.cat([x,available],1),'y':torch.tensor([c['emotion'] for c,z,v in rows]),'clip_ids':[c['clip_id'] for c,z,v in rows],'actor_ids':[c['actor_id'] for c,z,v in rows]}
    visual,vs=load_visual(device='cpu')
    frozen={'visual_loaded':sum(p.numel() for p in visual.parameters()),'visual_preclassifier':sum(p.numel() for n,p in visual.named_parameters() if not n.startswith('fc.')),'audio_loaded':sum(p.numel() for p in m.parameters()),'audio_preclassifier':sum(p.numel() for n,p in m.named_parameters() if not n.startswith('head.'))}
    torch.save(packed,CACHE/'pooled.pt');write_json(R/'feature_cache.json',{'status':'PASS','path':str(CACHE/'pooled.pt'),'sha256':file_hash(CACHE/'pooled.pt'),'raw_identity_hash':manifest['raw_identity_hash'],'audio_encoder_sha256':s['sha256'],'cal_read':False,'pooling':'valid-frame temporal mean; original indices/alignment retained','normalization':'train clip means only','frozen_parameters':frozen})
    print('FROZEN FEATURE CACHE PASS',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['visual','finish']);args=parser.parse_args()
    visual_stream() if args.stage=='visual' else finish()
