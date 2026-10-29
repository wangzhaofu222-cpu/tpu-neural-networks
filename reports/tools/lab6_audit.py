"""Independent numerical/provenance audit, with no model retraining."""
import json,math,hashlib,csv,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch,yaml
from scipy.signal import lfilter
from src.utils.io import write_json,file_hash,digest
from src.utils.preregistration import verify_git_hypothesis
from src.generative.distributions import distribution,make_roles
from src.generative.experiments import source_model
from src.generative.samplers import langevin_sample
from src.generative.density import density_grid
from src.diagnostics.density_metrics import bandwidth,kernel_mean,sample_metrics,paired_statistics,select_flow_depth
from src.diagnostics.invertibility import inverse_error,autograd_logdet
from src.analysis.generative_comparison import nearest_centers

R=Path('experiments/lab6')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    torch.set_num_threads(2);checks=[]
    def check(name,condition,detail=None):checks.append({'check':name,'status':'PASS' if bool(condition) else 'FAIL','detail':detail})
    protected=read('experiments/lab5_verification/historical_files.json')
    changed=[p for p,h in protected.items() if not Path(p).exists() or file_hash(p)!=h]
    check('Lab0_5_historical_bytes_protected',not changed,changed)
    check('Lab5_verification130tests625checks',read('experiments/lab5_verification/audit.json')['audit_checks']==625)
    old=read('experiments/lab5/final_audit.json');check('historical_states_retained',old['scientific_predictions']['E2_preregistered_magnitude_band'].startswith('FAIL') and not old['teacher_split_independently_verified'] and old['Lab4_Q2'].startswith('BLOCKED'))
    idx=read(R/'run_index.json');e1=read(R/'E1.json');e2=read(R/'E2.json');e3=read(R/'E3.json');lock=read(R/'E3_protocol_lock.json');prereg=read(R/'preregistration_evidence.json')
    check('real_immutable_E3_Git_hypothesis',verify_git_hypothesis(prereg['commit'],'experiments/lab6/hypothesis.md'))
    prereg_time=datetime.fromisoformat(prereg['committed_at'])
    sid=read('.private/lab4_student_id.json')['student_id'];h=hashlib.sha256((sid+':lab6').encode()).hexdigest();profile=read(R/'profile.json')
    expected={'lab6_target':['eight_gaussians','four_corners','three_blobs'][int(h[:2],16)%3],
        'lab6_eta':[[.01,.05,.1,.4],[.005,.02,.08,.5]][int(h[2:4],16)%2],'lab6_scale':[1.,2.,4.][int(h[4:6],16)%3]}
    check('private_profile_independent_hash_slices',all(profile['profile'][k]==v for k,v in expected.items()) and profile['lab6_hash']==h)
    protocol=read(R/'data_protocol.json');roles_cache={};metric_cache={}
    for target_name,data in protocol.items():
        roles=make_roles(target_name);roles_cache[target_name]=roles
        check('data_'+target_name,all(hashlib.sha256(x.numpy().tobytes()).hexdigest()==data['role_tensor_sha256'][role] and torch.equal(x,torch.load(Path('data/lab6')/target_name/(role+'.pt'),weights_only=False)) for role,x in roles.items()) and not data['cal_created'])
    for target_name in ['two_blobs','eight_gaussians']:
        p=read(R/'metrics_protocol'/f'{target_name}.json');roles=roles_cache[target_name];h,indices=bandwidth(roles['train']);self_kernel=kernel_mean(roles['test'],roles['test'],h)
        check('train_only_fixed_bandwidth_'+target_name,h==p['bandwidth'] and indices.tolist()==p['train_indices'] and abs(self_kernel-p['fixed_test_self_kernel_mean'])<1e-12)
        metric_cache[target_name]=(h,self_kernel)
        noise=read(R/'noise'/f'{target_name}.json');values=[]
        for r in noise['repetitions']:
            sample=torch.load(Path('runs/lab6')/r['RUN_ID']/'fresh_target_samples.pt',weights_only=False)
            target=distribution(target_name);met=sample_metrics(sample,roles['test'],target,h,self_kernel)
            check('noise_'+target_name+'_'+str(r['seed']),torch.equal(sample,distribution(target_name,r['seed']).sample(5000)) and abs(met['MMD2']-r['MMD2'])<1e-12 and met['Dmode']==r['Dmode'])
            values.append(met)
        check('five_noise_means_'+target_name,len(values)==5 and abs(sum(v['MMD2'] for v in values)/5-noise['noise_MMD2'])<1e-12 and abs(sum(v['Dmode'] for v in values)/5-noise['noise_Dmode'])<1e-12)
    print('Data/noise recalculation complete',flush=True)
    inventory=[];training_seconds=0.;formal_paths=set(idx['training'].values())
    check('eleven_distinct_full_model_training_runs',len(formal_paths)==11)
    for key,source in idx['training'].items():
        p=Path(source);c=yaml.safe_load((p/'config.resolved.yaml').read_text(encoding='utf-8'));m=read(p/'manifest.json');s=read(p/'summary.json')
        steps=[json.loads(x) for x in (p/'steps.jsonl').read_text().splitlines()]
        check('full5000_'+key,m['status']=='COMPLETED' and m['global_step']==5000 and len(steps)==5000 and [r['global_step'] for r in steps]==list(range(1,5001)) and not m['git_dirty'])
        check('fixed_optimizer_batch_'+key,c['optimizer']['name']=='adam' and c['optimizer']['lr']==.001 and c['train']['batch_size']==256 and s['samples_processed']==1280000 and not any(r['optimizer_step_skipped'] for r in steps))
        if c['model']['name']=='energy_model':
            check('EBM_training_sampler_'+key,all(r['sigma_squared']==2*c['sampling']['eta'] and len(r['energy_curve'])==51 and r['finite'] and r['energy_regularization']==0 for r in steps))
        for f in sorted((p/'checkpoints').glob('*.pt')):
            state=torch.load(f,weights_only=False,map_location='cpu');required={'model','optimizer','scheduler','scaler','python_rng','numpy_rng','torch_rng','cuda_rng','loader_rng','generative_state','config_hash','source_hash'}
            good=required<=state.keys() and state['config_hash']==digest(c) and state['source_hash']==m['source_hash'] and state['data_fingerprint']==m['data_fingerprint']
            if c['model']['name']=='energy_model':
                b=state['generative_state']['replay'];good=good and b['points'].shape==(10000,2) and b['generator_state'].dtype==torch.uint8 and b['restart']==.05
            if f.name in ['best.pt','last.pt']:good=good and state['global_step']==5000
            check('checkpoint_'+key+'_'+f.stem,good);inventory.append({'path':str(f),'sha256':file_hash(f),'RUN_ID':p.name,'global_step':state['global_step']})
        if key.startswith('E3_'):
            check('E3_prereg_before_'+key,datetime.fromisoformat(m['started_at'])>prereg_time)
            kind=key.split('_')[1];seed=c['experiment']['seed'];check('E3_locked_config_'+key,digest(c)==lock['config_hashes'][f'{kind}_{seed}'])
        training_seconds+=s['training_seconds']
    print('Training/checkpoint verification complete',flush=True)
    evaluated={};val_pointwise={}
    for key,r in idx['evaluations'].items():
        source=Path('runs/lab6')/r['source_RUN_ID'];model,c,state=source_model(source);p=Path('runs/lab6')/r['RUN_ID'];target=distribution(r['target']);roles=roles_cache[target.name]
        samples=torch.load(p/'generated_samples.pt',weights_only=False);h,self_kernel=metric_cache[target.name]
        met=sample_metrics(samples,roles['test'],target,h,self_kernel)
        check('sample_metrics_'+key,len(samples)==5000 and all(abs(met[k]-r[k])<1e-10 for k in ['Cov','Dmode','MMD2']))
        gi,gd=density_grid(model,target,r['kind']);saved_grid=torch.load(p/'density_grid.pt',weights_only=False)
        check('density_grid_'+key,torch.equal(gd['points'],saved_grid['points']) and torch.equal(gd['model_log_density'],saved_grid['model_log_density']) and abs(gi['Dgrid']-r['Dgrid'])<1e-12 and gi['normalized_on_omega']==r['normalized_on_omega'])
        gen=torch.Generator().manual_seed(r['sample_seed'])
        if r['kind']=='ebm':
            initial=torch.randn(5000,2,generator=gen);eta=r['sampler']['eta']
            end,traj,diag=langevin_sample(model,initial,eta,1.,50,target.domain,gen,True)
            saved=torch.load(p/'sampling_trajectory.pt',weights_only=False)
            check('full_frozen_sampler_reproduction_'+key,torch.equal(end,samples) and torch.equal(traj,saved) and diag['DK']==r['sampler']['DK'] and diag['pre_reflection_outside_proposal_fraction']==r['sampler']['pre_reflection_outside_proposal_fraction'])
            check('EBM_grid_not_exact_NLL_'+key,r['normalized_on_omega'] and 'exact_test_NLL' not in r and r['density_kind']=='grid_estimate_not_exact_likelihood')
        else:
            with torch.no_grad():fresh=model.sample(5000,gen);vnll=-model.log_prob(roles['val']).double();tnll=-model.log_prob(roles['test']).double()
            check('Flow_exact_likelihood_and_generation_'+key,torch.equal(fresh,samples) and abs(float(vnll.mean())-r['val_NLL'])<1e-10 and abs(float(tnll.mean())-r['exact_test_NLL'])<1e-10)
            double=copy_model_double(model)
            with torch.no_grad():err=inverse_error(double,roles['test'].double())
            jac=autograd_logdet(double,roles['test'][:100].double())
            check('Flow_inverse_acceptance_'+key,float(err.median())<=1e-5 and jac['max_error']<=1e-5,{'median':float(err.median()),'logdet_error':jac['max_error']})
            saved=torch.load(p/'invertibility.pt',weights_only=False)
            with torch.no_grad():far=inverse_error(double,saved['far_points'])
            check('Flow_farfield_recomputed_'+key,torch.equal(err,saved['epsilon']) and torch.equal(far,saved['far_epsilon']))
            if key.startswith('E2_'):val_pointwise[int(key.split('_')[-1])]=vnll
        mid=torch.tensor(r['midpoint']['point'],dtype=torch.float32)[None]
        with torch.no_grad():value=float(model.log_prob(mid)) if r['kind']=='flow' else float(-model(mid)-r['log_Z_hat_omega'])
        check('midpoint_'+key,abs(value-r['midpoint']['model_log_density'])<1e-7)
        evaluated[key]=met
        print('Evaluation recomputed',key,flush=True)
    for key,r in idx['quadratic'].items():
        stored=torch.load(Path('runs/lab6')/r['RUN_ID']/'quadratic_chains.pt',weights_only=False)
        traj=stored['trajectory'].numpy();noise=stored['standard_normal_innovations'].numpy()
        expected=lfilter([r['sigma']],[1.,-(1-r['eta'])],noise,axis=1);variance=float(traj[:,10001:].var(axis=1,ddof=1).mean())
        theory=r['sigma']**2/(2*r['eta']-r['eta']**2)
        check('full_quadratic_finite_chain_'+key,noise.shape==(8,210000) and traj.shape==(8,210001) and np.array_equal(expected,traj[:,1:]) and abs(variance-r['measured_variance'])<1e-12)
        check('quadratic_5percent_acceptance_'+key,abs(variance-theory)/theory<=.05)
    candidates=[r for r in e1['eta_rows'] if r['stable'] and r['center_to_other_mode_fraction']>=.2]
    selected=min(candidates,key=lambda r:r['eta']) if candidates else min([r for r in e1['eta_rows'] if r['stable']],key=lambda r:r['MMD2'])
    check('E1_eta_rule',selected['eta']==e1['selected_eta'] and bool(candidates)==e1['mixing_rule_satisfied'])
    check('E1_centers_below_midpoint',max(e1['center_energies'])<e1['origin_energy'],{'centers':e1['center_energies'],'midpoint':e1['origin_energy']})
    check('E1_tail_and_maxeta_theory',e1['tail_bound']<=.01 and e1['quadratic'][-1]['theory_variance']>=1.2)
    check('one_frozen_EBM_all_eta',len({idx['evaluations'][f"E1_eta_{r['eta']}"]['checkpoint_sha256'] for r in e1['eta_rows']})==1)
    selection=select_flow_depth(e2['rows'],val_pointwise)
    check('E2_val_only_depth_rule',selection==e2['selection'])
    for seed in [1,2,3]:
        a=Path(idx['training'][f'E3_ebm_{seed}']);b=Path(idx['training'][f'E3_flow_{seed}']);am=read(a/'manifest.json');bm=read(b/'manifest.json')
        ac=torch.load(a/'checkpoints/last.pt',weights_only=False);bc=torch.load(b/'checkpoints/last.pt',weights_only=False)
        check('E3_paired_data_order_'+str(seed),am['first_batch_checksum']==bm['first_batch_checksum'] and torch.equal(ac['loader_rng'],bc['loader_rng']) and am['data_fingerprint']==bm['data_fingerprint'])
        check('E3_capacity_20percent_'+str(seed),abs(am['parameters']-bm['parameters'])/min(am['parameters'],bm['parameters'])<=.2)
    q=e3['primary_metric'];noise=read(R/'noise/eight_gaussians.json')['noise_'+q]
    paired=paired_statistics([r['ebm'][q] for r in e3['paired_runs']],[r['flow'][q] for r in e3['paired_runs']],noise)
    check('E3_independent_paired_statistics',paired==e3['paired_statistics'])
    import xml.etree.ElementTree as ET
    tests=ET.parse('docs/environment/tests-lab6-final.xml').getroot().find('testsuite')
    check('all_final_tests',int(tests.attrib['failures'])==0 and int(tests.attrib['errors'])==0)
    answers=read(R/'counterfactual_answers.json');check('five_measured_counterfactuals',len(answers)==5 and all('explanation' in a for a in answers))
    qa=read(R/'figure_QA.json');check('all_figures_visually_reviewed',qa['status']=='PASS' and len(qa['figures'])==8)
    fp=read(R/'figure_provenance.json');check('figure_provenance_hashes',all(file_hash(R/'figures'/n)==h for n,h in fp['figure_hashes'].items()))
    finished=datetime.now(timezone.utc);first=min(read(Path(p)/'manifest.json')['started_at'] for p in formal_paths)
    oldstart=ET.parse('docs/environment/tests-lab5-verification-before-lab6.xml').getroot().find('testsuite').attrib['timestamp']
    stats={'formal_model_training_runs':len(formal_paths),'formal_optimizer_steps':5000*len(formal_paths),'formal_training_seconds':training_seconds,
        'formal_checkpoints':len(inventory),'all_lab6_checkpoints':len(list(Path('runs/lab6').glob('*/checkpoints/*.pt'))),
        'budget_training_runs':len(read(R/'budget.json')['measurements']),'formal_evaluation_runs':len(idx['evaluations']),
        'quadratic_runs':len(idx['quadratic']),'noise_runs':10,'final_tests_passed':int(tests.attrib['tests']),
        'first_formal_model_started_at':first,'audit_finished_at':finished.isoformat(),
        'formal_workflow_wall_seconds':(finished-datetime.fromisoformat(first)).total_seconds(),
        'verification_to_audit_wall_seconds':(finished-datetime.fromisoformat(oldstart)).total_seconds()}
    write_json(R/'checkpoint_inventory.json',inventory);write_json(R/'execution_statistics.json',stats)
    audit={'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL','checks':checks,'check_count':len(checks),
        'failed':[c for c in checks if c['status']=='FAIL'],'statistics':stats,
        'Lab4_Q2':'BLOCKED','Lab5_course_split':'UNVERIFIED','historical_E2_magnitude_prediction':'FAIL retained',
        'E1_mixing_observation':'PASS' if e1['mixing_rule_satisfied'] else 'FAIL observed; teacher-required fallback applied',
        'E3_statistical_conclusion':paired['conclusion'],'E3_hypothesis_supported':e3['hypothesis_supported'],
        'scope':'No formal report; no Lab7; no old model retraining'}
    write_json(R/'final_audit.json',audit);print(json.dumps({'status':audit['status'],'failed':audit['failed'],'statistics':stats},ensure_ascii=False))

def copy_model_double(model):
    import copy
    return copy.deepcopy(model).cpu().double()
if __name__=='__main__':main()
