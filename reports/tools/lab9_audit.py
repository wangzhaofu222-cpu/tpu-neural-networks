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
        if c['role']=='cal':check(cid+':cal not decoded',c['media_path'] is None and not (D/'preprocessed'/cid).exists() and not (D/'features/visual'/(cid+'.pt')).exists() and not (D/'features/audio'/(cid+'.pt')).exists());continue
        v=torch.load(D/'features/visual'/(cid+'.pt'),weights_only=False);a=torch.load(D/'features/audio'/(cid+'.pt'),weights_only=False)
        check(cid+':cache identity',v['clip_id']==a['clip_id']==cid and v['role']==a['role']==c['role'])
        check(cid+':frozen encoder identity',v['encoder_sha256']==read(R/'encoder_sources.json')['visual']['sha256'] and a['encoder_sha256']==read(R/'encoder_sources.json')['audio']['sha256'])
        check(cid+':normalization/frame cache identity',a['normalization_sha256']==file_hash(R/'mel_normalization.json') and v['input_frames_sha256']==c['video_frames_sha256'] and v['source_sha256']==c['source_sha256'])
        check(cid+':finite cache',torch.isfinite(v['z']).all() and torch.isfinite(a['z']).all())
        check(cid+':face skip/index',len(v['z'])==len(v['indices'])==c['valid5fps_frames'] and len(set(v['indices']))==len(v['indices']))
        alignment=torch.load(D/'features'/(cid+'_alignment.pt'),weights_only=False);assigned=alignment['assigned_indices'];check(cid+':no alignment overlap',len(assigned)==len(torch.unique(assigned)))
        times=alignment['audio_sample_times'];n=len(alignment['canonical_video_indices']);inside=(2*times>=-3200)&(2*(times-(n-1)*3200)<3200)
        check(cid+':no internal canonical alignment gaps',set(assigned.tolist())==set(torch.where(inside)[0].tolist()))
        check(cid+':frame validity retains original indices',alignment['visual_original_indices']==v['indices'] and int(alignment['visual_valid'].sum())==len(v['z']))
    stats=read(R/'mel_normalization.json');train=[c for c in manifest['clips'] if c['role']=='train'];check('mel stats train IDs only',set(stats['clip_ids'])=={c['clip_id'] for c in train})
    x=torch.cat([torch.load(c['log_mel_path'],weights_only=True).double() for c in train],1);check('mel train recomputation mean',np.allclose(x.mean(1).numpy(),stats['mean'],atol=1e-10));check('mel train recomputation std',np.allclose(x.std(1,unbiased=False).numpy(),stats['std'],atol=1e-9))
    del x
    f=torch.load(FEATURE,weights_only=False);check('feature normalization train mean',float(f['train']['x'][:,:640].mean(0).abs().max())<1e-4)
    from src.data.multimodal_data import clip_features
    maximum_api_error=0.
    for role in ['train','val','test']:
        for i,cid in enumerate(f[role]['clip_ids']):
            inp=clip_features(cid);v=inp['visual'][inp['align']['visual_valid']].mean(0);a=inp['audio'][inp['align']['audio_valid']].mean(0);actual=torch.cat([v,a]);error=float((actual-f[role]['x'][i,:640]).abs().max());maximum_api_error=max(maximum_api_error,error);check(cid+':single clip API equals fixed pooled input',error<1e-4)
    write_json(R/'single_clip_api_consistency.json',{'status':'PASS' if maximum_api_error<1e-4 else 'FAIL','clips':1320,'maximum_absolute_feature_difference':maximum_api_error,'tolerance':1e-4,'scope':'Float32 temporal mean/normalization order; all invalid canonical slots excluded. Packed training path and saved model inputs unchanged.','pooled_feature_sha256':file_hash(FEATURE)})
    registry=read(R/'run_registry.json');check('unique RUN IDs',len(registry)==len({r['RUN_ID'] for r in registry}));checkpoint_count=0;updates=0;training_seconds=0.
    prereg=read(R/'E1_preregistration.json');blob=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show',prereg['commit']+':experiments/lab9/hypothesis.md']);check('real prereg Git text unchanged (Windows newline normalization only)',blob.decode('utf-8').replace('\r\n','\n')==(R/'hypothesis.md').read_text(encoding='utf-8'))
    for row in registry:
        path=Path(row['path']);passport=read(path/'manifest.json');check(row['RUN_ID']+':passport',passport['RUN_ID']==row['RUN_ID'] and passport['status'].startswith('COMPLETED') and not passport['git_dirty'])
        check(row['RUN_ID']+':cal never read',passport['cal_read'] is False)
        if row.get('evaluation'):
            if not (path/'predictions.pt').exists():check(row['RUN_ID']+':raw saved',False);continue
            raw=torch.load(path/'predictions.pt',weights_only=False);results=read(path/'results.json')
            ecfg=yaml.safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8'));spec=ecfg['evaluation_spec'];capacity=sum(read(Path(spec[k])/'manifest.json')['trainable_parameters'] for k in ['video_path','audio_path']) if spec.get('strategy')=='late' else read(Path(spec['path'])/'manifest.json')['trainable_parameters']
            check(row['RUN_ID']+':complete source head capacity',passport['parameters']==capacity);check(row['RUN_ID']+':summary head capacity',results['fusion_profile']['parameters']==capacity)
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
            if cfg['model']['name']=='multimodal_fusion':check(row['RUN_ID']+':fixed full head budget',cfg['train']['epochs']==passport['completed_epoch']==200 and passport['global_step']==2800)
            for p in (path/'checkpoints').glob('*.pt'):
                checkpoint_count+=1;s=torch.load(p,map_location='cpu',weights_only=False);model.load_state_dict(s['model']);check(str(p)+':state identity',s['run_id']==row['RUN_ID'] and s['config_hash']==passport['config_hash'] and s['best_metric_name']=='macro_f1')
                check(str(p)+':complete resume state',all(k in s for k in ['optimizer','scheduler','scaler','python_rng','numpy_rng','torch_rng','loader_rng','cuda_rng']))
            if 'main' in row['kind']:
                rc=subprocess.run(['git','-c','safe.directory=D:/神经网络','merge-base','--is-ancestor',prereg['commit'],passport['git_commit']]).returncode;check(row['RUN_ID']+':after prereg ancestor',rc==0)
    e1=read(R/'E1_main.json');delta=np.array([r['F']['macro_f1']-r['U']['macro_f1'] for r in e1['rows']]);check('paired delta recomputed',np.allclose(delta,e1['differences'],atol=1e-12));check('paired sample SD',abs(delta.std(ddof=1)-e1['sample_std'])<1e-12);check('all three seeds',[r['seed'] for r in e1['rows']]==[1,2,3])
    check('paired mean',abs(delta.mean()-e1['mean'])<1e-12);est=bool(((delta>0).all() or (delta<0).all()) and abs(delta.mean())>=2*delta.std(ddof=1));check('teacher paired decision without threshold changes',e1['decision']==('различие установлено' if est else 'различие не установлено'))
    from reports.tools.lab9_common import predict,training_capacity
    selection=read(R/'E1_validation_selection.json');validation_probabilities={};label=f['val']['y'].numpy()
    for name,spec in selection['specs'].items():
        pp,_=predict(spec,f['val']['x']);validation_probabilities[name]=pp
        check(name+':locked validation F1 reproduced',abs(classification(pp.numpy(),label)['macro_f1']-selection['validation'][name])<1e-12)
        if spec['strategy']!='late':
            cfg=yaml.safe_load((Path(spec['path'])/'config.resolved.yaml').read_text(encoding='utf-8'));check(name+':architecture capacity',training_capacity(build_model(cfg['model']))['total_trainable']==selection['parameter_counts'][name]['total_trainable'])
    torch.save({'probabilities':validation_probabilities,'labels':f['val']['y'],'clip_ids':f['val']['clip_ids'],'generation':'Independent post-experiment deterministic replay; original locked metrics remain unchanged in preregistration Git','source_preregistration_commit':prereg['commit']},R/'E1_validation_probability_replay.pt')
    for row in selection['late_weight_grid']:
        pp=row['weight']*validation_probabilities['video']+(1-row['weight'])*validation_probabilities['audio'];check('late lambda '+str(row['weight'])+':val grid replay',abs(classification(pp.numpy(),label)['macro_f1']-row['macro_f1'])<1e-12)
    for fusion,controls in selection['controls'].items():
        target=selection['parameter_counts'][fusion]['total_trainable']
        for modality,control in controls.items():check(fusion+':'+modality+':capacity within1percent or nearest integer',control['match']['relative_difference']<=.01 and control['match']['target']==target)
    for p in (D/'features/corrupt').glob('snr_*.pt'):
        obj=torch.load(p,weights_only=False);check(p.name+':all perclip SNR',len(obj['records'])==300 and max(abs(r['target_snr_db']-r['measured_snr_db']) for r in obj['records'])<=.1)
    check('five measured counterfactuals',len(read(R/'counterfactuals.json')['answers'])==5)
    check('actual CPU/CUDA inference agreement',read(R/'cpu_gpu_consistency.json')['status']=='PASS')
    check('registered late API bitwise equals original functional predictions',read(R/'late_registry_equivalence.json')['status']=='PASS')
    check('canonical metadata repair retained identical learned inputs',read(R/'alignment_metadata_audit.json')['pooled_features_sha256_before']==file_hash(FEATURE))
    expected=read(R/'figures/artifact_hashes.json')
    from reports.tools.lab9_plots import main as plots
    replay=R/'audit_replayed_figures';plots(replay)
    for name,h in expected.items():check('regenerated:'+name,file_hash(replay/name)==h)
    failed=[r for r in checks if r['status']=='FAIL'];out={'status':'PASS' if not failed else 'FAIL','audit_checks':len(checks),'passed':len(checks)-len(failed),'failed':failed,'checks':checks,'checkpoint_count':checkpoint_count,'formal_training_count':sum(not r.get('evaluation',False) for r in registry),'evaluation_runs':sum(r.get('evaluation',False) for r in registry),'optimizer_updates':updates,'training_wall_seconds':training_seconds,'history_files_preserved':len(old),'teacher_split':'UNVERIFIED','encoder_profile_conflict':'UNVERIFIED'}
    write_json(R/'audit.json',out);print('AUDIT',out['status'],len(checks),'failed',len(failed),flush=True)
    from reports.tools.lab9_audit_metrics import main as metric_audit
    metric_audit()

if __name__=='__main__':main()
