import json
import numpy as np,torch
from scipy.io import wavfile
from reports.tools.lab9_common import *
from reports.tools.lab9_features import load_audio,audio_features,load_visual,visual_features
from src.multimodal.corruptions import add_noise,time_mask,frame_dropout
from src.multimodal.audio import log_mel
from src.multimodal.pairs import derangement

def corrupted_inputs(kind,level):
    cache=D/'features/corrupt';cache.mkdir(parents=True,exist_ok=True);target=cache/f'{kind}_{level}.pt'
    encoders=read(R/'encoder_sources.json');signature={'audio':encoders['audio']['sha256'],'visual':encoders['visual']['sha256'],'normalization':file_hash(R/'feature_normalization.json'),'clean_features':file_hash(FEATURE),'raw_data':read(R/'data_manifest.json')['raw_identity_hash']}
    if target.exists():
        cached=torch.load(target,weights_only=False)
        if cached.get('cache_signature')!=signature:raise ValueError('Corrupt cache encoder/data/normalization identity changed; do not reuse stale features')
        return cached
    f=features();x=f['test']['x'].clone();ids=f['test']['clip_ids'];clips={c['clip_id']:c for c in read(R/'data_manifest.json')['clips']};stats=read(R/'feature_normalization.json');mean=torch.tensor(stats['mean']);std=torch.tensor(stats['std']);records=[]
    if kind in ['snr','time_mask']:m,s,melstats=load_audio()
    if kind=='blur':m,s=load_visual()
    for i,cid in enumerate(ids):
        c=clips[cid];meta={'clip_id':cid,'kind':kind,'level':level,'encoder_sha256':s['sha256'] if kind!='frame_dropout' else read(R/'encoder_sources.json')['visual']['sha256']}
        if kind in ['snr','time_mask']:
            if kind=='snr':
                fs,wave=wavfile.read(c['audio_path']);assert fs==16000;y,measured=add_noise(wave,level,seed=1001+i);mel=log_mel(y);meta.update(measured)
            else:mel=time_mask(torch.load(c['log_mel_path'],weights_only=True),int(level),seed=1001+i)
            z=audio_features(mel,m,melstats);x[i,512:640]=(z.mean(0)-mean[512:])/std[512:]
        else:
            if kind=='blur':z=visual_features(torch.load(c['video_frames_path'],weights_only=True)['frames'],m,s,blur=float(level))
            elif kind=='frame_dropout':
                z=torch.load(D/'features/visual'/(cid+'.pt'),weights_only=False)['z'];valid=frame_dropout(torch.ones(len(z),dtype=torch.bool),int(level),seed=1001+i);z=z[valid];meta['retained_frames']=len(z)
            else:raise ValueError(kind)
            x[i,:512]=(z.mean(0)-mean[:512])/std[:512] if len(z) else 0;x[i,640]=float(len(z)>0)
        records.append(meta)
    out={'x':x,'clip_ids':ids,'records':records,'kind':kind,'level':level,'cache_signature':signature,'feature_normalization_sha256':file_hash(R/'feature_normalization.json'),'input_role':'test','cal_read':False};torch.save(out,target)
    return out

def main():
    exploration=read(R/'E1_exploration.json');specs=exploration['selection']['specs'];drop={'path':train('E2_gated_modality_dropout','gated',p_drop=config()['profile']['lab9_p_drop']),'strategy':'gated'};specs={**{k:specs[k] for k in ['early','gated','late05','late_val']},'gated_dropout':drop}
    results={k:evaluate('E2_modes_'+k,spec) for k,spec in specs.items()}
    f=features();x=f['test']['x'];y=f['test']['y'];perm=derangement(len(x),seed=1);shuffled=x.clone();shuffled[:,512:640]=x[perm,512:640];spec=specs['gated'];shuffle=evaluate('E2_audio_shuffle',spec,x=shuffled,intervene=False)
    original,_=predict(spec,x);sp,_=predict(spec,shuffled);masked,_=predict(spec,x,(1,0))
    shuffle.update(permutation=perm.tolist(),fixed_points=int((perm==np.arange(len(x))).sum()),labels_unchanged=True,mean_delta_shuffle=float((sp-original).abs().sum(1).mean()),mean_delta_mask=float((masked-original).abs().sum(1).mean()),changed_class_shuffle=float((sp.argmax(1)!=original.argmax(1)).float().mean()),changed_class_mask=float((masked.argmax(1)!=original.argmax(1)).float().mean()))
    noise={k:[] for k in ['gated','gated_dropout']}
    for snr in [30,20,10,0,-5]:
        c=corrupted_inputs('snr',snr)
        assert max(abs(a['measured_snr_db']-snr) for a in c['records'])<=.1
        for k in noise:
            r=evaluate(f'E2_SNR{snr}_{k}',specs[k],x=c['x']);r['snr_db']=snr;r['corruption_cache']=str(D/'features/corrupt'/f'snr_{snr}.pt');noise[k].append(r)
    thresholds={k:next((r['snr_db'] for r in noise[k] if r['macro_f1']<results[k]['contribution']['M_without_audio']),None) for k in noise}
    supplementary={}
    for kind,level in [('time_mask',1),('frame_dropout',1),('blur',.5)]:
        c=corrupted_inputs(kind,level);supplementary[kind]=evaluate('E2_'+kind,specs['gated'],x=c['x'])
    baseline=exploration['test'];diff=baseline['video']['macro_f1']-baseline['audio']['macro_f1']
    write_json(R/'E2.json',{'models':results,'shuffle':shuffle,'noise':noise,'highest_grid_snr_worse_than_missing':thresholds,'absence_meaning':'None means no tested SNR was worse; no interpolation/extrapolation','supplementary':supplementary,'MV_minus_MA_separately_trained':diff,'same_model_Rv_minus_Ra':{k:r['contribution']['Rv']-r['contribution']['Ra'] for k,r in results.items()},'blur_protocol':'Lab8 blur level1 sigma=.5, kernel5 on raw224face tensor before same frozen encoder','specs':specs})

if __name__=='__main__':main()
