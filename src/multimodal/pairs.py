import numpy as np


def derangement(n,seed=1):
    if n<2:raise ValueError('Need at least two clips for shuffle')
    rng=np.random.default_rng(seed);order=rng.permutation(n);shift=int(rng.integers(1,n));p=np.empty(n,dtype=int);p[order]=np.roll(order,shift)
    assert len(set(p))==n and np.all(p!=np.arange(n));return p


def make_pairs(clips,protocol,count=300,seed=1):
    if protocol not in ['A','B']:raise ValueError('Unknown pairing protocol')
    possible=[(i,j) for i,a in enumerate(clips) for j,b in enumerate(clips) if a['actor_id']!=b['actor_id'] and ((a['emotion']==b['emotion'])==(protocol=='A'))]
    if len(possible)<count:raise ValueError('Insufficient different-actor pairs; cannot fabricate samples')
    indices=np.random.default_rng(seed).choice(len(possible),count,replace=False)
    return [{'video_index':possible[k][0],'audio_index':possible[k][1],'video_clip_id':clips[possible[k][0]]['clip_id'],'audio_clip_id':clips[possible[k][1]]['clip_id'],
             'video_actor_id':clips[possible[k][0]]['actor_id'],'audio_actor_id':clips[possible[k][1]]['actor_id'],'video_emotion':clips[possible[k][0]]['emotion'],'audio_emotion':clips[possible[k][1]]['emotion'],'protocol':protocol,'seed':seed} for k in indices]


def require_defined_metric(protocol,metric):
    if protocol=='B' and metric.lower() in ['accuracy','macro_f1','f1','ece','risk','selective_risk']:
        raise ValueError(f'{metric} undefined for conflicting Protocol B: no unique true emotion label')
