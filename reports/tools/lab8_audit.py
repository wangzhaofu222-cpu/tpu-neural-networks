"""Independent NumPy metric oracle, role/provenance/attack and history checks."""
import json,subprocess,time
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
import torch,yaml
from reports.tools.lab8_common import R,read,registry,outputs,role_data,source_model,config
from src.utils.io import file_hash,write_json,digest
from src.utils.preregistration import verify_git_hypothesis
from src.utils.provenance import git_state

def oracle(z,y,T=1.,M=15):
    a=np.asarray(z,dtype=np.float64)/T;a-=a.max(1,keepdims=True);lp=a-np.log(np.exp(a).sum(1,keepdims=True));p=np.exp(lp);c=p.max(1);correct=p.argmax(1)==np.asarray(y);bins=np.minimum((c*M).astype(int),M-1)
    e=sum(np.sum(bins==i)/len(y)*abs(correct[bins==i].mean()-c[bins==i].mean()) for i in range(M) if np.any(bins==i))
    return {'accuracy':float(correct.mean()),'mean_confidence':float(c.mean()),'ece':float(e),'nll':float(-lp[np.arange(len(y)),y].mean()),'brier':float(((p-np.eye(p.shape[1])[y])**2).sum(1).mean())}

def main():
    from src.utils.reproducibility import seed_all
    start=time.perf_counter();seed_all(1,True,2);checks=[];inventory=[]
    def check(name,ok,detail=None):checks.append({'check':name,'status':'PASS' if bool(ok) else 'FAIL','detail':detail})
    old=read('experiments/lab7_verification/audit.json');check('Lab7_180_tests_467_checks_39_checkpoints',old['status']=='PASS' and old['audit_checks']==467 and old['checkpoints']==39)
    oldhash=read('experiments/lab7_verification/historical_files.json');changed=[p for p,h in oldhash.items() if not Path(p).exists() or file_hash(p)!=h];check('historical_experiment_data_and_models_immutable',not changed,changed)
    changed_source=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','diff','--name-status','ef1d2fa636afc1f36f3223c6a8d78e1cbe498edf','--','src','tests','scripts','configs']).decode().splitlines()
    check('historical_source_unmodified',all(s.startswith('A\t') for s in changed_source),changed_source)
    check('Lab0_7_directories_exist',all(Path(f'experiments/lab{i}').is_dir() for i in range(8)))
    check('shared_harness_preserved',subprocess.check_output(['git','-c','safe.directory=D:/神经网络','diff','ef1d2fa636afc1f36f3223c6a8d78e1cbe498edf','--','src/engine/runner.py','src/utils/provenance.py'])==b'')
    cfg=config();profile=read(R/'profile.json');import hashlib
    h=hashlib.sha256((read('.private/lab4_student_id.json')['student_id']+':lab8').encode()).hexdigest()
    expected=(['cnn','vit'][int(h[:2],16)%2],['gaussian_noise','gaussian_blur','contrast','occlusion'][int(h[2:4],16)%4],[2,4,8][int(h[4:6],16)%3])
    check('individual_profile_independent_SHA256',(profile['lab8_arch'],profile['lab8_corruption'],profile['lab8_pgd_eps_255'])==expected)
    split=read('splits/cifar10.json');roles=split['roles'];check('train_val_cal_disjoint',all(not(set(roles[a])&set(roles[b])) for a,b in [('train','val'),('train','cal'),('val','cal')]))
    check('test_separate_official_source',len(roles['test'])==10000 and len(roles['val'])==5000)
    check('course_protocol_UNVERIFIED_preserved',read(R/'data_protocol.json')['course_protocol_status']=='UNVERIFIED')
    check('cal_not_read',read(R/'data_protocol.json')['cal'].startswith('not accessed'))
    source=read(R/'model_sources.json')
    from src.models.small_cnn import build_model
    for key,row in source.items():
        p=Path(row['path']);state=torch.load(p,map_location='cpu',weights_only=False);model=build_model(row['cfg']['model']);model.load_state_dict(state['model'],strict=True)
        check('checkpoint_'+key,file_hash(p)==row['sha256'] and state['run_id']==row['RUN_ID'] and state['global_step']==10000 and state['config_hash']==digest(row['cfg']))
        check('controlled_training_protocol_'+key,row['fraction']==.05 and row['steps']==10000 and read(p.parent.parent/'manifest.json')['seed']==row['seed'] and row['cfg']['model']['classes']==10)
        inventory.append({'source_key':key,'path':str(p),'RUN_ID':row['RUN_ID'],'sha256':file_hash(p),'seed':row['seed'],'global_step':state['global_step']})
    check('three_distinct_real_CNN_seeds',len({source['cnn_'+str(i)]['RUN_ID'] for i in [1,2,3]})==3)
    check('E2_all_training_conditions_fixed',all(source['cnn_1']['cfg'][field]==source[f'cnn_{i}']['cfg'][field] for field in ['model','optimizer','scheduler','data','train','profile'] for i in [2,3]))
    check('equal_CNN_train_subsets',len({read(Path(source['cnn_'+str(i)]['path']).parent.parent/'manifest.json')['subset_fingerprint'] for i in [1,2,3]})==1)
    from src.data.vision import datasets
    _,current_info=datasets(source['cnn_1']['cfg'])
    for sk,row in source.items():
        original=read(Path(row['path']).parent.parent/'manifest.json')
        check('train_only_normalization_and_dataset_'+sk,original['normalization']==current_info['normalization'] and original['data_fingerprint']==current_info['data_fingerprint'] and original['split_fingerprint']==current_info['split_fingerprint'])
    pre=read(R/'preregistration_evidence.json');check('E2_real_Git_hypothesis',verify_git_hypothesis(pre['commit'],'experiments/lab8/hypothesis.md'))
    lock=read(R/'E2_protocol_lock.json');check('E1_evidence_locked',file_hash(R/'E1.json')==lock['E1_sha256'])
    blob=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show',pre['commit']+':experiments/lab8/E2_protocol_lock.json']);check('E2_lock_immutable',blob.replace(b'\r\n',b'\n')==(R/'E2_protocol_lock.json').read_bytes().replace(b'\r\n',b'\n'))
    exploratory='b0636ba71e077a3db32b6620fca10351f8f0a238';check('E1_E3_prospective_Git',verify_git_hypothesis(exploratory,'experiments/lab8/exploratory_hypotheses.md'))
    preregtime=datetime.fromisoformat(pre['committed_at']);expltime=datetime.fromisoformat(subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show','-s','--format=%cI',exploratory]).decode().strip());current=git_state()
    idx=registry();check('RUN_ID_unique',len(set(idx.values()))==len(idx))
    all_output=[];total_wall=0;formal=0
    for key,path in idx.items():
        p=Path(path);m=read(p/'manifest.json');s=read(p/'summary.json');c=yaml.safe_load((p/'config.resolved.yaml').read_text(encoding='utf-8'))
        check('passport_'+key,m['RUN_ID']==p.name and m['config_hash']==digest(c) and m['status']=='COMPLETED_EVALUATION' and m['optimizer_updates']==0)
        data_companion=read(p/'data_protocol.resolved.json');actual_data=read(R/'data_protocol.json')
        check('explicit_image_data_provenance_'+key,data_companion['manifest_original_sha256']==file_hash(p/'manifest.json') and data_companion['actual_CIFAR10_raw_file_sha256']==actual_data['CIFAR10_raw_file_sha256'] and data_companion['actual_SVHN_test_sha256']==file_hash('data/svhn/test_32x32.mat') and data_companion['split_sha256']==file_hash('splits/cifar10.json') and data_companion['cal_role_accessed'] is False)
        if m['purpose']=='formal':
            formal+=1;total_wall+=m.get('wall_seconds',0)
            check('committed_code_'+key,not m['git_dirty'] and m['source_hash']==current['source_hash'])
            check('prospective_hypothesis_'+key,datetime.fromisoformat(m['started_at'])>expltime)
            if key.startswith(('corruption_','T5_')) or key in ['clean_val_cnn_2','clean_test_cnn_2','clean_val_cnn_3','clean_test_cnn_3']:
                check('E2_prereg_before_'+key,datetime.fromisoformat(m['started_at'])>preregtime)
        if not (p/'outputs.pt').exists():continue
        d=outputs(p);ids=d['object_IDs'];role=d['role'];x,y,expectedids=role_data(role);N=len(ids)
        check('role_IDs_'+key,ids==expectedids[:N] and (d['labels'] is None if role=='ood' else torch.equal(d['labels'],y[:N])))
        check('finite_logits_'+key,d['logits'].shape==(N,10) and torch.isfinite(d['logits']).all())
        if d['labels'] is not None:
            o=oracle(d['logits'],d['labels']);stored=s['metrics'];check('independent_numpy_metrics_'+key,max(abs(o[k]-(stored['ece']['value'] if k=='ece' else stored[k])) for k in o)<1e-12)
        else:check('OOD_metrics_undefined_'+key,s['metrics'] is None)
        z=[];attack_legal=True;restarts_legal=True
        for f in sorted((p/'batches').glob('*.pt')):
            part=torch.load(f,weights_only=False);z.append(part['logits'])
            if 'adversarial_pixels' in part:
                begin=int(f.stem);adv=part['adversarial_pixels'];raw=x[begin:begin+len(adv)].float()/255;eps=c['evaluation']['attack']['epsilon']
                attack_legal &= bool((adv-raw).abs().max()<=eps+1e-6 and adv.min()>=0 and adv.max()<=1)
                if 'restart_losses' in part.get('attack_info',{}):
                    info=part['attack_info'];restarts_legal &= torch.equal(info['selected_losses'],info['restart_losses'].max(0).values)
        check('durable_shards_'+key,torch.equal(torch.cat(z),d['logits']))
        if c['evaluation'].get('attack'):
            check('actual_attack_pixels_'+key,attack_legal);check('worst_restart_per_object_'+key,restarts_legal)
            check('every_attack_step_constraints_'+key,all(t['linf_max']<=c['evaluation']['attack']['epsilon']+1e-6 and 0<=t['pixel_min']<=t['pixel_max']<=1 for t in s['constraints']))
        for f in p.glob('records_*.jsonl'):
            count=0;valid=True
            with f.open(encoding='utf-8') as stream:
                for line in stream:
                    r=json.loads(line);valid &= r['object_ID']==ids[count] and r['OOD']==(role=='ood') and (r['correct'] is None if role=='ood' else isinstance(r['correct'],bool));count+=1
            check('object_records_'+key+'_'+f.stem,valid and count==N)
        all_output.append({'path':str(p/'outputs.pt'),'bytes':(p/'outputs.pt').stat().st_size,'sha256':file_hash(p/'outputs.pt'),'RUN_ID':p.name})
    temps=read(R/'temperatures.json')
    for key,fit in temps.items():
        p=Path(idx['clean_val_'+key]);d=outputs(p);before=oracle(d['logits'],d['labels']);after=oracle(d['logits'],d['labels'],fit['temperature'])
        check('val_only_temperature_'+key,fit['val_IDs']==role_data('val')[2] and fit['role']=='val' and fit['test_used'] is False)
        check('val_NLL_nonincrease_'+key,after['nll']<=before['nll'] and abs(after['nll']-fit['val_nll_after'])<1e-12)
    e1=read(R/'E1.json');e2=read(R/'E2.json');e3=read(R/'E3.json');ss=read(R/'selective_ood.json')
    for arch in ['cnn','vit']:
        rr=[r for r in e1['rows'] if r['model']==arch];d=outputs(idx['clean_test_'+arch+'_1']);T=rr[1]['temperature']
        check('E1_argmax_100percent_'+arch,torch.equal(d['logits'].argmax(1),(d['logits']/T).argmax(1)) and rr[0]['accuracy']==rr[1]['accuracy'])
        check('E1_bin_sensitivity_'+arch,len([r for r in e1['binning_sensitivity'] if r['model']==arch])==6)
        model,source_row=source_model(arch+'_1');x,_,_=role_data('test');raw=x[:64].cuda().float()/255
        with torch.no_grad():a=model(raw);b=model.model((raw-model.mean)/model.std)
        check('historical_normalization_logits_'+arch,torch.equal(a,b))
        import copy
        cpu_model=copy.deepcopy(model).cpu().eval()
        with torch.no_grad():cpu_logits=cpu_model(raw.cpu())
        check('actual_CPU_GPU_model_consistency_'+arch,torch.allclose(a.cpu(),cpu_logits,atol=1e-4,rtol=1e-5) and torch.equal(a.cpu().argmax(1),cpu_logits.argmax(1)),{'maximum_logit_error':float((a.cpu()-cpu_logits).abs().max()),'N':64})
        # Saved inference agrees with the historical Lab2 final test evaluation, when indexed.
        historic=read('experiments/lab2/evaluations.json');historic_rows=historic.values() if isinstance(historic,dict) else historic
        candidates=[]
        for q in historic_rows:
            if isinstance(q,dict) and q.get('parent_RUN_ID')==source_row['RUN_ID'] and q.get('corruption','clean')=='clean':candidates.append(q)
        observed=oracle(outputs(idx['clean_test_'+arch+'_1'])['logits'],outputs(idx['clean_test_'+arch+'_1'])['labels'])
        check('historical_Lab2_test_metrics_'+arch,bool(candidates) and all(abs(q['accuracy']-observed['accuracy'])<1e-12 and abs(q['loss']-observed['nll'])<1e-6 and q['checkpoint_sha256']==source_row['sha256'] for q in candidates))
        check('source_validation_metric_reproduced_'+arch,abs(oracle(outputs(idx['clean_val_'+arch+'_1'])['logits'],outputs(idx['clean_val_'+arch+'_1'])['labels'])['accuracy']-source_row['source_summary']['validation_metric'])<1e-12)
    check('E2_all_five_types_six_levels',len([r for r in e2['rows'] if r['seed']==1])==60 and {r['corruption'] for r in e2['rows']}==set(cfg['corruptions']))
    cleanrows=[r for r in e1['rows'] if r['model']=='cnn']
    for row in [r for r in e2['rows'] if r['level']==0]:
        expectedrow=next(r for r in cleanrows if r['temperature']==row['temperature']);check('E2_level0_'+row['corruption']+'_'+str(row['temperature']),all(abs(row[k]-expectedrow[k])<=1e-6 for k in ['accuracy','nll','brier','mean_confidence']) and abs(row['ece']['value']-expectedrow['ece']['value'])<=1e-6)
    deltas=np.array([r['gA']-r['gB'] for r in e2['pairs']]);check('E2_paired_mean_sd',np.allclose(deltas,e2['statistics']['deltas'],rtol=0,atol=1e-15) and abs(deltas.mean()-e2['statistics']['mean'])<1e-15 and abs(deltas.std(ddof=1)-e2['statistics']['sample_sd'])<1e-15)
    check('E2_each_seed_own_T',len({temps[f'cnn_{i}']['RUN_ID'] for i in [1,2,3]})==3)
    check('T5_fit_val5_only',e2['T5']['fit']['val_IDs']==role_data('val')[2] and e2['T5']['fit']['val_nll_after']<=e2['T5']['fit']['val_nll_before'])
    for arch in ['cnn','vit']:
        rows=[r for r in e3['rows'] if r['source_key']==arch+'_1'];check('E3_complete_grid_'+arch,len(rows)==8 and sorted(r['eps_255'] for r in rows if r['attack']=='FGSM')==[0,1,2,4,8])
        ids=[outputs(Path('runs/lab8')/r['RUN_ID'])['object_IDs'] for r in rows];check('E3_identical_full_test_'+arch,all(x==role_data('test')[2] for x in ids))
        fg=next(r for r in rows if r['attack']=='FGSM' and r['eps_255']==2);pg=next(r for r in rows if r['attack']=='PGD' and r['K']==10);zero=next(r for r in rows if r['attack']=='FGSM' and r['eps_255']==0)
        check('E3_PGD_not_weaker_'+arch,pg['metrics'][0]['accuracy']<=fg['metrics'][0]['accuracy'])
        check('E3_epsilon0_identity_accuracy_'+arch,zero['metrics'][0]['accuracy']==next(r for r in e1['rows'] if r['model']==arch)['accuracy'])
        check('E3_PGD20_R2_real_'+arch,any(r['attack']=='PGD' and r['K']==20 and r['R']==2 for r in rows))
    for s in ss:
        check('SVHN_full_test_'+s['source_key'],outputs(idx['OOD_'+s['source_key']])['logits'].shape[0]==26032)
        for c in s['conditions']:
            from scipy.stats import rankdata
            td=outputs(idx['clean_test_'+s['source_key']]);od=outputs(idx['OOD_'+s['source_key']])
            def measures(z):
                z=np.asarray(z,dtype=np.float64)/c['temperature'];z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);top=np.sort(p,axis=1)[:,-2:]
                return {'msp':1-top[:,-1],'entropy':-np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0),axis=1),'margin':1-top[:,-1]+top[:,-2]}
            ui=measures(td['logits']);uo=measures(od['logits'])
            for measure,v in c['measures'].items():
                sr=v['selective'];error=1-oracle(outputs(idx['clean_test_'+s['source_key']])['logits'],outputs(idx['clean_test_'+s['source_key']])['labels'])['accuracy']
                check('selective_'+s['source_key']+'_'+measure+'_'+str(c['temperature']),abs(sr['risk'][-1]-error)<1e-12 and sr['oracle_AURC']<=sr['AURC'] and abs(np.mean(sr['risk'])-sr['AURC'])<1e-12)
                # Independent tie-aware rank-sum AUROC and threshold-count AP/FPR oracle.
                a=ui[measure];b=uo[measure];n=len(a);q=len(b);ranks=rankdata(np.r_[a,b],method='average');roc=(ranks[n:].sum()-q*(q+1)/2)/(n*q)
                thresholds=np.unique(np.r_[a,b])[::-1];tp=q-np.searchsorted(np.sort(b),thresholds,side='left');fp=n-np.searchsorted(np.sort(a),thresholds,side='left');recall=tp/q
                ap=np.sum(np.diff(np.r_[0.,recall])*tp/(tp+fp));fpr=fp[np.flatnonzero(recall>=.95)[0]]/n
                # Near-equal floating ties may differ only at machine precision between NumPy/Torch.
                check('independent_OOD_metrics_'+s['source_key']+'_'+measure+'_'+str(c['temperature']),abs(roc-v['ood']['AUROC'])<2e-6 and abs(ap-v['ood']['AUPR'])<2e-6 and abs(fpr-v['ood']['FPR95'])<1e-4)
    for sk in ['cnn_1','cnn_2','cnn_3','vit_1']:
        p=read(R/'profiles'/sk/'reliability.json');check('reliability_profile_'+sk,{'clean','corruption','adversarial','selective','ood','undefined'}<=p.keys() and len(p['undefined'])>=4)
    suite=ET.parse('docs/environment/tests-lab8-final.xml').getroot().find('testsuite');check('all_tests_actually_pass',int(suite.attrib['tests'])>=239 and int(suite.attrib['failures'])==int(suite.attrib['errors'])==0)
    check('smoke_resume_verified',read(R/'budget.json')['interrupted_shard_resume_verified'])
    check('five_measured_counterfactuals',len(read(R/'counterfactual_answers.json'))==5 and all(a['RUN_IDs'] for a in read(R/'counterfactual_answers.json')))
    check('artifact_regeneration',read(R/'regeneration_audit.json')['status']=='PASS');check('visual_QA',read(R/'figure_QA.json')['status']=='PASS')
    provenance=read(R/'figure_provenance.json');qa=read(R/'figure_QA.json')
    check('figure_hashes_and_review_coverage',all(file_hash(R/'figures'/n)==h for n,h in provenance['figure_hashes'].items()) and len(qa['figures'])==provenance['figure_groups'])
    check('no_formal_report_modified',subprocess.check_output(['git','-c','safe.directory=D:/神经网络','diff','ef1d2fa636afc1f36f3223c6a8d78e1cbe498edf','--','*.tex','*.pdf'])==b'')
    check('no_Lab9_execution',not Path('experiments/lab9').exists())
    write_json(R/'checkpoint_inventory.json',inventory)
    all_raw=list(Path('runs/lab8').glob('*/outputs.pt'))+list(Path('runs/lab8').glob('*/batches/*.pt'))+list(Path('runs/lab8').glob('*/records_*.jsonl'))
    write_json(R/'raw_artifact_inventory.json',[{'path':str(p),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(all_raw)])
    statistics={'tests_passed':int(suite.attrib['tests']),'Lab7_audit_checks':467,'checks':len(checks),'formal_evaluation_runs':formal,'smoke_runs':len(idx)-formal,'classifier_training_runs':0,'new_checkpoints':0,'source_checkpoints_verified':len(inventory),'formal_run_seconds_sum':total_wall,'audit_seconds':time.perf_counter()-start,'all_run_folders':len(idx),'figure_groups':len(list((R/'figures').glob('*.png')))}
    write_json(R/'final_audit.json',{'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL','checks':checks,'failed':[c for c in checks if c['status']=='FAIL'],'check_count':len(checks),'statistics':statistics,'historical_states':read(R/'requirement_status.json')['historical_states']})
    print('AUDIT',len(checks),'checks, FAIL',[c['check'] for c in checks if c['status']=='FAIL'],flush=True)

if __name__=='__main__':main()
