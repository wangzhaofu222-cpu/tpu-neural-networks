"""Supplemental independent definitions: CM/gains, waveform SNR and paired AUROC."""
from pathlib import Path
import json,math
import numpy as np,torch,yaml
from scipy.io import wavfile
from reports.tools.lab9_common import R,D,read,features
from src.utils.io import write_json,file_hash,digest

def classification(probabilities,labels):
    # Independent boolean-count implementation, not the training metric function.
    y=np.asarray(labels);pred=np.asarray(probabilities).argmax(1);scores=[]
    for c in range(8):
        tp=np.count_nonzero((pred==c)&(y==c));fp=np.count_nonzero((pred==c)&(y!=c));fn=np.count_nonzero((pred!=c)&(y==c));den=2*tp+fp+fn;scores.append(2*tp/den if den else 0.)
    cm=[[int(np.count_nonzero((y==i)&(pred==j))) for j in range(8)] for i in range(8)]
    return {'macro_f1':float(np.mean(scores)),'accuracy':float(np.count_nonzero(y==pred)/len(y)),'confusion_matrix':cm}

def main():
    checks=[]
    def check(name,condition):checks.append({'name':'metrics:'+name,'status':'PASS' if bool(condition) else 'FAIL','detail':None})
    registry=read(R/'run_registry.json');recomputed={}
    source=read(R/'encoder_sources.json');ds=features();majority=int(np.bincount(ds['train']['y'].numpy(),minlength=8).argmax())
    for row in registry:
        path=Path(row['path']);passport=read(path/'manifest.json');cfg=yaml.safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8'));check(row['RUN_ID']+':physical resolved config hash',digest(cfg)==passport['config_hash'])
        protocol=passport['multimodal_protocol'];check(row['RUN_ID']+':explicit timing and encoder passport',protocol['eta_v_fps']==5 and protocol['fs']==16000 and protocol['n_hop']==160 and protocol['r']==1 and protocol['tau_w_seconds']==.2 and protocol['av_offset_ms']==0 and (protocol.get('produced_audio_checkpoint_sha256')==source['audio']['sha256'] if row['kind']=='AudioCNN_train_val_selected' else protocol['frozen_visual_checkpoint_sha256']==source['visual']['sha256'] and protocol['frozen_audio_checkpoint_sha256']==source['audio']['sha256']))
        if not row.get('evaluation'):continue
        raw=torch.load(path/'predictions.pt',weights_only=False);out=read(path/'results.json')
        if 'probabilities' in raw:
            metric=classification(raw['probabilities'].numpy(),raw['labels'].numpy());recomputed[row['RUN_ID']]=metric
            check(row['RUN_ID']+':independent boolean-count macroF1',abs(metric['macro_f1']-out['macro_f1'])<1e-12)
            if 'no_audio' in raw:
                full=metric['macro_f1'];na=classification(raw['no_audio'].numpy(),raw['labels'].numpy())['macro_f1'];nv=classification(raw['no_video'].numpy(),raw['labels'].numpy())['macro_f1'];bp=np.eye(8)[np.full(len(raw['labels']),majority)];m0=classification(bp,raw['labels'].numpy())['macro_f1'];den=full-m0
                for key,expected in [('Ra',full-na),('Rv',full-nv),('Rtilde_a',(full-na)/den if den>0 else None),('Rtilde_v',(full-nv)/den if den>0 else None)]:check(row['RUN_ID']+':independent '+key,out['contribution'][key] is None if expected is None else abs(out['contribution'][key]-expected)<1e-12)
            check(row['RUN_ID']+':confusion matrix',metric['confusion_matrix']==out['confusion_matrix']);check(row['RUN_ID']+':accuracy',abs(metric['accuracy']-out['accuracy'])<1e-12)
            check(row['RUN_ID']+':raw probability checksum',file_hash(path/'predictions.pt')==out['predictions_sha256'])
    e1=read(R/'E1_exploration.json');selection=e1['selection'];M={k:recomputed[r['RUN_ID']]['macro_f1'] for k,r in e1['test'].items()}
    for key,gains in e1['gains'].items():
        expected={'G':M[key]-max(M['video'],M['audio']),'Gint':M[key]-M['late_val'],'Gctrl':M[key]-max(M[selection['controls'][key][s]['name']] for s in ['video','audio'])}
        for metric,value in expected.items():check(key+':'+metric,abs(value-gains[metric])<1e-12)
    F=max(['early','gated','late05','late_val'],key=lambda k:selection['validation'][k]);U=max([selection['controls'][F][s]['name'] for s in ['video','audio']],key=lambda k:selection['validation'][k]);check('val-only F/U rankings',F==selection['F'] and U==selection['U'])
    source=read(R/'encoder_sources.json')
    for modality in ['visual','audio']:check(modality+':source checkpoint SHA256',file_hash(source[modality]['path'])==source[modality]['sha256'])
    e2=read(R/'E2.json');clips={c['clip_id']:c for c in read(R/'data_manifest.json')['clips']};waves={}
    for p in sorted((D/'features/corrupt').glob('snr_*.pt')):
        cache=torch.load(p,weights_only=False)
        for meta in cache['records']:
            cid=meta['clip_id']
            if cid not in waves:
                fs,x=wavfile.read(clips[cid]['audio_path']);assert fs==16000;waves[cid]=x.astype(np.float64)
            x=waves[cid];xi=torch.randn(x.shape,generator=torch.Generator().manual_seed(meta['seed']),dtype=torch.float64).numpy();signal=float(np.dot(x,x));noise=float(np.dot(xi,xi));gamma=math.sqrt(signal/(10**(meta['target_snr_db']/10)*noise));measured=10*math.log10(signal/(meta['gamma']**2*noise))
            check(p.stem+':'+cid+':independent waveform SNR',abs(measured-meta['target_snr_db'])<=.1);check(p.stem+':'+cid+':independent gamma',abs(gamma-meta['gamma'])<=1e-10*max(1.,gamma))
    for model,rows in e2['noise'].items():
        missing=e2['models'][model]['contribution']['M_without_audio'];highest=next((r['snr_db'] for r in rows if recomputed[r['RUN_ID']]['macro_f1']<missing),None);check(model+':highest discrete SNR',highest==e2['highest_grid_snr_worse_than_missing'][model])
    s=e2['shuffle'];permutation=np.asarray(s['permutation']);check('shuffle no fixed points',len(set(permutation))==300 and np.all(permutation!=np.arange(300)))
    original=torch.load(Path('runs/lab9')/e2['models']['gated']['RUN_ID']/'predictions.pt',weights_only=False);shuffled=torch.load(Path('runs/lab9')/s['RUN_ID']/'predictions.pt',weights_only=False)
    check('shuffle labels unchanged',torch.equal(original['labels'],shuffled['labels']));check('shuffle L1 recomputed',abs(float((original['probabilities']-shuffled['probabilities']).abs().sum(1).mean())-s['mean_delta_shuffle'])<1e-6)
    data=features()['test'];pairs=read(R/'E3_pairs.json')
    for protocol,rows in pairs.items():
        check(protocol+':unique ordered pairs',len({(r['video_clip_id'],r['audio_clip_id']) for r in rows})==len(rows)>=200)
        for i,r in enumerate(rows):
            v=clips[r['video_clip_id']];a=clips[r['audio_clip_id']];check(protocol+f':pair{i}:identity and class rule',v['actor_id']!=a['actor_id'] and ((v['emotion']==a['emotion'])==(protocol=='A')) and data['clip_ids'][r['video_index']]==v['clip_id'] and data['clip_ids'][r['audio_index']]==a['clip_id'])
    for name,out in read(R/'E3.json').items():
        raw=torch.load(Path('runs/lab9')/out['RUN_ID']/'predictions.pt',weights_only=False)
        for protocol in ['A','B']:
            rows=pairs[protocol];p=raw[protocol]['probabilities'];vi=torch.tensor([r['video_index'] for r in rows]);ai=torch.tensor([r['audio_index'] for r in rows]);yv=data['y'][vi];ya=data['y'][ai];pred=p.argmax(1);cond=out['conditions'][protocol]
            if protocol=='A':
                metric=classification(p.numpy(),yv.numpy());check(name+':A F1 and CM',abs(metric['macro_f1']-cond['macro_f1'])<1e-12 and metric['confusion_matrix']==cond['confusion_matrix'])
            else:
                for key,expected in [('rv',(pred==yv).float().mean()),('ra',(pred==ya).float().mean()),('r0',((pred!=yv)&(pred!=ya)).float().mean())]:check(name+':B following '+key,abs(float(expected)-cond[key])<1e-7)
            check(name+':'+protocol+':delta_ij',torch.allclose((p-raw['original_probabilities'][vi]).abs().sum(1),raw[protocol]['delta_ij'],atol=1e-7,rtol=0))
            check(name+':'+protocol+':delta_a',torch.allclose((p-raw[protocol]['no_audio_probabilities']).abs().sum(1),raw[protocol]['delta_a'],atol=1e-7,rtol=0))
        for key,a,b in [('AUROC_1_minus_conf',1-raw['A']['probabilities'].max(1).values.numpy(),1-raw['B']['probabilities'].max(1).values.numpy()),('AUROC_delta_a',raw['A']['delta_a'].numpy(),raw['B']['delta_a'].numpy())]:
            expected=float((b[:,None]>a[None,:]).mean()+.5*(b[:,None]==a[None,:]).mean());check(name+':direct pairwise probability '+key,abs(expected-out[key])<1e-12)
    from reports.tools.lab9_counterfactuals import main as replay_answers
    replay=R/'audit_replayed_answers';before=len(registry);replay_answers(reuse_double=True,destination=replay);check('answer replay performs no training/evaluation',len(read(R/'run_registry.json'))==before)
    for name in ['counterfactuals.json','counterfactuals.md']:check('deterministic answer regeneration '+name,file_hash(replay/name)==file_hash(R/name))
    audit=read(R/'audit.json');base=[c for c in audit['checks'] if not c['name'].startswith('metrics:')];audit['checks']=base+checks;audit['audit_checks']=len(audit['checks']);audit['failed']=[c for c in audit['checks'] if c['status']=='FAIL'];audit['passed']=audit['audit_checks']-len(audit['failed']);audit['status']='PASS' if not audit['failed'] else 'FAIL';audit['metric_supplement_checks']=len(checks);write_json(R/'audit.json',audit);write_json(R/'metric_audit.json',{'checks':checks,'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL'});print('METRIC AUDIT',len(checks),'COMBINED',audit['status'],audit['audit_checks'],'failed',len(audit['failed']))

if __name__=='__main__':main()
