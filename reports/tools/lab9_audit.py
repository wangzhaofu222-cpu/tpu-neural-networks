"""Independent checks against saved identities, checkpoints and raw predictions."""
import json,subprocess
from pathlib import Path
import numpy as np,torch,yaml
from src.utils.io import file_hash,write_json
from src.models.small_cnn import build_model
from src.diagnostics.contribution import classification,contribution,majority_baseline
from src.multimodal.identity import validate_clip_identity
from reports.tools.lab9_common import R,D,FEATURE,read
from reports.tools.lab9_e3 import auroc

def main():
    checks=[]
    def check(name,condition,detail=None):checks.append({'name':name,'status':'PASS' if bool(condition) else 'FAIL','detail':detail})
    old=read('experiments/lab8_verification/historical_files.json')
    for p,h in old.items():check('history:'+p,Path(p).is_file() and file_hash(p)==h)
    manifest=read(R/'data_manifest.json');split=read(R/'planned_actor_split.json')['roles'];check('actor/clip/frame identities',validate_clip_identity(manifest['clips'],split))
    check('official 1440',len(manifest['clips'])==1440);check('all actor archives verified',len(list((R/'acquisition').glob('actor_*.json')))==24)
    for p in (R/'acquisition').glob('actor_*.json'):
        a=read(p);check(p.name+':MD5',a['official_md5']==a['verified_md5']);check(p.name+':SHA256',file_hash(a['archive'])==a['sha256'])
    for c in manifest['clips']:
        cid=c['clip_id']
        if c['role']=='cal':check(cid+':cal not decoded',c['media_path'] is None and not (D/'preprocessed'/cid).exists());continue
        v=torch.load(D/'features/visual'/(cid+'.pt'),weights_only=False);a=torch.load(D/'features/audio'/(cid+'.pt'),weights_only=False)
        check(cid+':cache identity',v['clip_id']==a['clip_id']==cid and v['role']==a['role']==c['role'])
        check(cid+':frozen encoder identity',v['encoder_sha256']==read(R/'encoder_sources.json')['visual']['sha256'] and a['encoder_sha256']==read(R/'encoder_sources.json')['audio']['sha256'])
        check(cid+':finite cache',torch.isfinite(v['z']).all() and torch.isfinite(a['z']).all())
        check(cid+':face skip/index',len(v['z'])==len(v['indices'])==c['valid5fps_frames'] and len(set(v['indices']))==len(v['indices']))
        alignment=torch.load(D/'features'/(cid+'_alignment.pt'),weights_only=False);assigned=alignment['assigned_indices'];check(cid+':no alignment overlap',len(assigned)==len(torch.unique(assigned)))
    stats=read(R/'mel_normalization.json');train=[c for c in manifest['clips'] if c['role']=='train'];check('mel stats train IDs only',set(stats['clip_ids'])=={c['clip_id'] for c in train})
    x=torch.cat([torch.load(c['log_mel_path'],weights_only=True).double() for c in train],1);check('mel train recomputation mean',np.allclose(x.mean(1).numpy(),stats['mean'],atol=1e-10));check('mel train recomputation std',np.allclose(x.std(1,unbiased=False).numpy(),stats['std'],atol=1e-9))
    del x
    f=torch.load(FEATURE,weights_only=False);check('feature normalization train mean',float(f['train']['x'][:,:640].mean(0).abs().max())<1e-4)
    registry=read(R/'run_registry.json');check('unique RUN IDs',len(registry)==len({r['RUN_ID'] for r in registry}));checkpoint_count=0;updates=0;training_seconds=0.
    prereg=read(R/'E1_preregistration.json');blob=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show',prereg['commit']+':experiments/lab9/hypothesis.md']);check('real prereg Git text unchanged (Windows newline normalization only)',blob.decode('utf-8').replace('\r\n','\n')==(R/'hypothesis.md').read_text(encoding='utf-8'))
    for row in registry:
        path=Path(row['path']);passport=read(path/'manifest.json');check(row['RUN_ID']+':passport',passport['RUN_ID']==row['RUN_ID'] and passport['status'].startswith('COMPLETED') and not passport['git_dirty'])
        check(row['RUN_ID']+':cal never read',passport['cal_read'] is False)
        if row.get('evaluation'):
            if not (path/'predictions.pt').exists():check(row['RUN_ID']+':raw saved',False);continue
            raw=torch.load(path/'predictions.pt',weights_only=False);results=read(path/'results.json')
            if 'probabilities' in raw:
                p=raw['probabilities'].numpy();y=raw['labels'].numpy();check(row['RUN_ID']+':F1 recompute',abs(classification(p,y)['macro_f1']-results['macro_f1'])<1e-12)
                check(row['RUN_ID']+':probability simplex',np.allclose(p.sum(1),1,atol=1e-6) and (p>=0).all())
                if 'no_audio' in raw:
                    M0=majority_baseline(f['train']['y'].numpy(),y)['macro_f1'];cc=contribution(p,raw['no_audio'].numpy(),raw['no_video'].numpy(),y,M0)
                    for k in ['Ra','Rv','Rtilde_a','Rtilde_v']:
                        expected=cc[k];actual=results['contribution'][k];check(row['RUN_ID']+':'+k,expected==actual if expected is None else abs(expected-actual)<1e-12)
                    for k in ['delta_a','delta_v']:check(row['RUN_ID']+':'+k,np.allclose(cc[k],results['contribution'][k]))
            else:
                for protocol in ['A','B']:check(row['RUN_ID']+':pair N '+protocol,len(raw[protocol]['probabilities'])>=200)
                cond=results['conditions']['B'];check(row['RUN_ID']+':no undefined B metrics',not any(k in cond for k in ['accuracy','macro_f1','ece','risk']));check(row['RUN_ID']+':following sum',abs(sum(cond[k] for k in ['rv','ra','r0'])-1)<1e-6)
                labels=np.r_[np.zeros(len(raw['A']['probabilities'])),np.ones(len(raw['B']['probabilities']))]
                for k,score in [('AUROC_1_minus_conf',np.r_[1-raw['A']['probabilities'].max(1).values.numpy(),1-raw['B']['probabilities'].max(1).values.numpy()]),('AUROC_delta_a',np.r_[raw['A']['delta_a'].numpy(),raw['B']['delta_a'].numpy()])]:check(row['RUN_ID']+':'+k,abs(auroc(labels,score)-results[k])<1e-12)
        else:
            cfg=yaml.safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8'));model=build_model(cfg['model']);updates+=passport['global_step'];training_seconds+=passport['wall_seconds']
            for p in (path/'checkpoints').glob('*.pt'):
                checkpoint_count+=1;s=torch.load(p,map_location='cpu',weights_only=False);model.load_state_dict(s['model']);check(str(p)+':state identity',s['run_id']==row['RUN_ID'] and s['config_hash']==passport['config_hash'] and s['best_metric_name']=='macro_f1')
            if 'main' in row['kind']:
                rc=subprocess.run(['git','-c','safe.directory=D:/神经网络','merge-base','--is-ancestor',prereg['commit'],passport['git_commit']]).returncode;check(row['RUN_ID']+':after prereg ancestor',rc==0)
    e1=read(R/'E1_main.json');delta=np.array([r['F']['macro_f1']-r['U']['macro_f1'] for r in e1['rows']]);check('paired delta recomputed',np.allclose(delta,e1['differences'],atol=1e-12));check('paired sample SD',abs(delta.std(ddof=1)-e1['sample_std'])<1e-12);check('all three seeds',[r['seed'] for r in e1['rows']]==[1,2,3])
    for p in (D/'features/corrupt').glob('snr_*.pt'):
        obj=torch.load(p,weights_only=False);check(p.name+':all perclip SNR',len(obj['records'])==300 and max(abs(r['target_snr_db']-r['measured_snr_db']) for r in obj['records'])<=.1)
    check('five measured counterfactuals',len(read(R/'counterfactuals.json')['answers'])==5)
    expected=read(R/'figures/artifact_hashes.json')
    from reports.tools.lab9_plots import main as plots
    replay=R/'audit_replayed_figures';plots(replay)
    for name,h in expected.items():check('regenerated:'+name,file_hash(replay/name)==h)
    failed=[r for r in checks if r['status']=='FAIL'];out={'status':'PASS' if not failed else 'FAIL','audit_checks':len(checks),'passed':len(checks)-len(failed),'failed':failed,'checks':checks,'checkpoint_count':checkpoint_count,'formal_training_count':sum(not r.get('evaluation',False) for r in registry),'evaluation_runs':sum(r.get('evaluation',False) for r in registry),'optimizer_updates':updates,'training_wall_seconds':training_seconds,'history_files_preserved':len(old),'teacher_split':'UNVERIFIED','encoder_profile_conflict':'UNVERIFIED'}
    write_json(R/'audit.json',out);print('AUDIT',out['status'],len(checks),'failed',len(failed),flush=True)

if __name__=='__main__':main()
