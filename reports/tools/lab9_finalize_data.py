"""Audit genuine archive identities and compute train-only mel statistics."""
import json
from pathlib import Path
import torch
from src.utils.io import write_json,digest,file_hash
from src.multimodal.identity import validate_clip_identity

R=Path('experiments/lab9')
def main():
    assert (R/'acquisition_complete.json').is_file()
    assert (R/'preprocessing_complete.json').is_file()
    archives=[json.loads(p.read_text()) for p in sorted((R/'acquisition').glob('actor_*.json'))]
    assert len(archives)==24
    clips=[]
    for a in archives:
        assert a['official_md5']==a['verified_md5'] and len(a['clips'])==60
        for c in a['clips']:
            if c['role']=='cal':
                assert c['media_path'] is None
                assert not (Path('data/lab9/preprocessed')/c['clip_id']).exists()
                clips.append(c)
            else:
                m=json.loads((Path('data/lab9/preprocessed')/c['clip_id']/'metadata.json').read_text())
                assert m['clip_id']==c['clip_id'] and m['actor_id']==c['actor_id'] and m['role']==c['role']
                assert m['audio_pts_parse_found'] and m['audio_samples']>0
                assert file_hash(m['audio_path'])==m['audio_sha256']
                assert file_hash(m['video_frames_path'])==m['video_frames_sha256']
                clips.append(m)
    assert len({c['clip_id'] for c in clips})==1440
    validate_clip_identity(clips,json.loads((R/'planned_actor_split.json').read_text())['roles'])
    sizes={r:sum(c['role']==r for c in clips) for r in ['train','val','cal','test']}
    assert sizes==dict(train=840,val=180,cal=120,test=300)
    sums=torch.zeros(64,dtype=torch.float64);squares=sums.clone();n=0
    for c in clips:
        if c['role']!='train':continue
        x=torch.load(c['log_mel_path'],weights_only=True).double()
        assert x.shape[0]==64 and torch.isfinite(x).all()
        sums+=x.sum(1);squares+=x.square().sum(1);n+=x.shape[1]
    mean=sums/n;std=(squares/n-mean.square()).clamp_min(1e-12).sqrt()
    write_json(R/'mel_normalization.json',{'mean':mean.tolist(),'std':std.tolist(),'fit_role':'train','frames':n,'clip_ids':[c['clip_id'] for c in clips if c['role']=='train'],'ddof':0})
    manifest={'status':'PASS','clips':clips,'split_sizes':sizes,'cal_read':False,'teacher_split_status':'TEACHER_SPLIT_UNVERIFIED','raw_identity_hash':digest({a['actor_id']:a['sha256'] for a in archives}),'split_fingerprint':file_hash(R/'planned_actor_split.json'),'class_counts':{str(i):sum(c['emotion']==i for c in clips) for i in range(8)},'noncal_face_missing_clips':[c['clip_id'] for c in clips if c['role']!='cal' and c['valid5fps_frames']==0]}
    write_json(R/'data_manifest.json',manifest)
    print('GENUINE DATA AUDIT PASS',sizes,'missing faces',len(manifest['noncal_face_missing_clips']),flush=True)

if __name__=='__main__':main()
