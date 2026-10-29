"""Independent data/provenance/numerical audit; no optimizer updates."""
import json,math,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch,yaml
from src.utils.io import file_hash,digest,write_json
from src.utils.provenance import git_state
from src.utils.preregistration import verify_git_hypothesis
from src.models.small_cnn import build_model
from src.data.gmm2d import roles_and_stats,standardize,inverse_standardize,target,ExactPredictor
from src.diffusion.schedules import make_schedule,terminal_kl
from src.diffusion.samplers import ddpm_sample,ddim_sample
from src.diagnostics.density_metrics import kernel_mean
from src.diagnostics.sample_quality import quality,frequencies
from reports.tools.lab7_common import R,base,read,source_model


def main():
    torch.set_num_threads(2);checks=[];findings=[]
    def check(name,ok,detail=None):
        checks.append({'check':name,'status':'PASS' if bool(ok) else 'FAIL','detail':detail})
    protected=read('experiments/lab6_verification/historical_files.json')
    changed=[p for p,h in protected.items() if not Path(p).exists() or file_hash(p)!=h]
    check('Lab0_6_history_immutable',not changed,changed)
    old=read('experiments/lab6_verification/audit.json')
    check('Lab6_153_tests_234_audit_63_checkpoints',old['status']=='PASS' and old['audit_checks']==234 and old['all_checkpoints_loaded']==63)
    oldstates=read('experiments/lab6/requirement_status.json')['historical_states']
    check('historical_failures_not_rewritten',oldstates['Lab4Q2']=='BLOCKED' and oldstates['Lab5_course_split']=='UNVERIFIED' and not read('experiments/lab6/E1.json')['mixing_rule_satisfied'])
    cfg=base();profile=read(R/'profile.json')
    import hashlib
    sid=read('.private/lab4_student_id.json')['student_id'];h=hashlib.sha256((sid+':lab7').encode()).hexdigest()
    expected=([[500,.0001,.04],[1000,.0001,.02]][int(h[:2],16)%2],[[6,.1],[8,.08],[10,.06]][int(h[2:4],16)%3],[.01,.02,.04][int(h[4:6],16)%3])
    check('private_profile_hash_slices',(profile['lab7_T_beta'],profile['lab7_modes'],profile['lab7_rare_weight'])==expected)
    idx=read(R/'run_index.json');e1=read(R/'E1.json');e2=read(R/'E2.json');diag=read(R/'diagnostics.json');e3=read(R/'E3.json');sup=read(R/'supplements.json')
    lock=read(R/'E3_protocol_lock.json');prereg=read(R/'preregistration_evidence.json')
    check('immutable_real_Git_hypothesis',verify_git_hypothesis(prereg['commit'],'experiments/lab7/hypothesis.md'))
    blob=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show',prereg['commit']+':experiments/lab7/E3_protocol_lock.json'])
    check('immutable_Git_lock',blob.replace(b'\r\n',b'\n')==(R/'E3_protocol_lock.json').read_bytes().replace(b'\r\n',b'\n'))
    check('E1_E2_selections_unchanged',file_hash(R/'E1.json')==lock['E1_sha256'] and file_hash(R/'E2.json')==lock['E2_sha256'])
    exploratory='7eb93b2a66e67a1d4135d9a4a3ae8cffb288e04f'
    check('exploratory_hypothesis_realGit',verify_git_hypothesis(exploratory,'experiments/lab7/exploratory_hypotheses.md'))
    prospective=datetime.fromisoformat(subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show','-s','--format=%cI',exploratory]).decode().strip())
    prereg_time=datetime.fromisoformat(prereg['committed_at']);code=git_state();inventory=[];trainseconds=0.
    for key,path in idx['training'].items():
        p=Path(path);c=yaml.safe_load((p/'config.resolved.yaml').read_text(encoding='utf-8'));m=read(p/'manifest.json');summary=read(p/'summary.json')
        steps=[json.loads(l) for l in (p/'steps.jsonl').read_text().splitlines()]
        check('full20000_'+key,m['status']=='COMPLETED' and m['global_step']==20000 and len(steps)==20000 and [r['global_step'] for r in steps]==list(range(1,20001)))
        check('formal_budget_'+key,c['train']['batch_size']==512 and c['optimizer']['lr']==.001 and c['optimizer']['name']=='adam' and summary['samples_processed']==10240000 and not any(r['optimizer_step_skipped'] for r in steps))
        check('clean_source_'+key,not m['git_dirty'] and m['source_hash']==code['source_hash'])
        check('RUN_config_seed_identity_'+key,m['RUN_ID']==p.name and m['config_hash']==digest(c) and m['seed']==c['experiment']['seed'] and c['experiment']['role']==key)
        roles,stats=roles_and_stats(c);hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in roles.items()}
        check('data_and_train_only_stats_'+key,hashes==m['role_fingerprints'] and m['normalization']==stats and m['split_sizes']['cal']==0)
        z=standardize(roles['train'],stats)
        check('unit_train_variance_'+key,torch.allclose(z.mean(0),torch.zeros(2,dtype=torch.float64),atol=1e-12) and torch.allclose(z.var(0,correction=0),torch.ones(2,dtype=torch.float64),atol=1e-12))
        schedule=make_schedule(c['diffusion']);check('valid_terminal_'+key,schedule.alpha_bar[-1]<=1e-3 and abs(float(schedule.alpha_bar[-1])-c['diffusion']['alpha_bar_T'])<1e-15)
        resolved=read(p/'schedule.resolved.json')
        check('actual_schedule_beta_range_'+key,resolved['effective_beta_range']==[float(schedule.beta[1:].min()),float(schedule.beta[1:].max())] and resolved['original_config_sha256']==file_hash(p/'config.resolved.yaml'))
        check('hypothesis_before_training_'+key,datetime.fromisoformat(m['started_at'])>(prereg_time if key.startswith('E3_') else prospective))
        for f in sorted((p/'checkpoints').glob('*.pt')):
            state=torch.load(f,map_location='cpu',weights_only=False);model=build_model(c['model']);model.load_state_dict(state['model'],strict=True)
            required={'model','optimizer','scheduler','scaler','python_rng','numpy_rng','torch_rng','cuda_rng','loader_rng','generative_state'}
            check('checkpoint_'+key+'_'+f.stem,required<=state.keys() and state['config_hash']==digest(c) and state['source_hash']==m['source_hash'] and state['run_id']==p.name and 'training_noise_rng' in state['generative_state'] and all(torch.isfinite(v).all() for v in state['model'].values()))
            inventory.append({'path':str(f),'sha256':file_hash(f),'RUN_ID':p.name,'seed':c['experiment']['seed'],'global_step':state['global_step'],'purpose':'formal'})
        trainseconds+=summary['training_seconds']
    print('Training and checkpoints audited',flush=True)
    # PairedE1 streams are identical despite schedule-specific losses.
    ca=Path(idx['training']['E1_linear']);cb=Path(idx['training']['E1_cosine'])
    aa=[json.loads(l) for l in (ca/'steps.jsonl').read_text().splitlines()];bb=[json.loads(l) for l in (cb/'steps.jsonl').read_text().splitlines()]
    check('E1_paired_training_t_epsilon_stream',all(a['t_mean']==b['t_mean'] and a['noise_checksum_sum']==b['noise_checksum_sum'] for a,b in zip(aa,bb)))
    ac=yaml.safe_load((ca/'config.resolved.yaml').read_text());bc=yaml.safe_load((cb/'config.resolved.yaml').read_text());ac['experiment']['role']=bc['experiment']['role'];ac['diffusion']=bc['diffusion']
    check('E1_only_schedule_differs',ac==bc)
    short=make_schedule({**cfg['diffusion'],'T':cfg['diffusion']['T']//10},diagnostic=True)
    base_roles,base_stats=roles_and_stats(cfg)
    check('short_T_diagnostic_actual_terminal',not short.protocol_valid and abs(float(short.alpha_bar[-1])-e1['boundary_short']['alpha_bar_T'])<1e-15 and abs(terminal_kl(short,standardize(base_roles['train'],base_stats))-e1['boundary_short']['terminal_KL'])<1e-15)
    protocols={}
    for path in sorted((R/'metrics').glob('*.json')):
        if path.name=='bandwidth.json':continue
        protocol=read(path);c=base();c['data']['pi0']=protocol['pi0'];dist=target(c);roles,stats=roles_and_stats(c)
        refsamples=torch.load(R/'metrics'/str(protocol['pi0'])/'reference100000.pt',weights_only=False)
        reference=frequencies(refsamples,dist)
        check('reference100k_'+path.stem,torch.equal(refsamples,dist.sample(100000,704)) and reference.tolist()==protocol['reference_frequencies'])
        for row in protocol['noise']:
            samples=torch.load(Path('runs/lab7')/row['RUN_ID']/'generated_samples.pt',weights_only=False)
            q=quality(samples,roles['test'],dist,reference,protocol['h'],protocol['test_self_kernel'])
            check('noise_'+path.stem+'_'+str(row['seed']),torch.equal(samples,dist.sample(5000,row['seed'])) and all(abs(q[k]-row[k])<1e-12 for k in ['MMD2','Dmode','Cov']))
        for metric in ['MMD2','Dmode']:
            vals=np.array([r[metric] for r in protocol['noise']])
            check('noise_mean_SD_'+path.stem+'_'+metric,abs(vals.mean()-protocol[metric+'_noise_mean'])<1e-12 and abs(vals.std(ddof=1)-protocol[metric+'_noise_sd'])<1e-12)
        protocols[protocol['pi0']]=(protocol,roles,stats)
    for key,row in idx['evaluations'].items():
        p=Path('runs/lab7')/row['RUN_ID'];c=yaml.safe_load((p/'config.resolved.yaml').read_text());protocol,roles,stats=protocols[c['data']['pi0']]
        samples=torch.load(p/'generated_samples.pt',weights_only=False);std=torch.load(p/'standardized_samples.pt',weights_only=False)
        q=quality(samples,roles['test'],target(c),protocol['reference_frequencies'],protocol['h'],protocol['test_self_kernel'])
        check('metrics_'+key,len(samples)==5000 and all(abs(q[k]-row[k])<1e-12 for k in ['MMD2','Dmode','Cov','rare_frequency']))
        check('original_units_'+key,torch.equal(inverse_standardize(std.double(),stats),samples))
        xT=torch.load(p/'initial_noise.pt',weights_only=False)
        expected=torch.randn(5000,2,generator=torch.Generator().manual_seed(row['initial_seed']),dtype=torch.float64).to(xT.dtype)
        check('fixed_initial_noise_'+key,torch.equal(xT,expected))
        check('NFE_tau_no_clipping_'+key,row['NFE']==row['K']==len(row['tau']) and row['tau'][-1]==c['diffusion']['T'] and not row['clipping'] and all(a<b for a,b in zip(row['tau'],row['tau'][1:])))
        check('timing_repetitions_'+key,len(row['seconds_repetitions'])==row['timing_repeats'] and row['repeat_max_difference']==0 and abs(float(np.median(row['seconds_repetitions']))-row['seconds_median'])<1e-12)
        check('source_checkpoint_hash_'+key,file_hash(Path('runs/lab7')/row['source_RUN_ID']/'checkpoints/last.pt')==row['source_checkpoint_sha256'])
        resolved=read(p/'schedule.resolved.json');check('resolved_reverse_protocol_'+key,resolved['tau']==row['tau'] and resolved['DDIM_eta']==row['eta'] and resolved['original_config_sha256']==file_hash(p/'config.resolved.yaml'))
        # Replay every saved sampler from the actual model/exact predictor and RNG seed.
        model,_,_=source_model(Path('runs/lab7')/row['source_RUN_ID']);device=c['runtime']['device'];s=make_schedule(c['diffusion'],diagnostic=not row['terminal_protocol_valid']).to(device)
        fn=model if row['predictor']=='network' else ExactPredictor(target(c),stats,s) if row['predictor']=='exact' else lambda x,t:torch.zeros_like(x)
        g=torch.Generator(device=device).manual_seed(row['noise_seed']);xt=xT.to(device)
        if row['sampler']=='ddpm':out,_=ddpm_sample(fn,s,xt,generator=g)
        else:out,_=ddim_sample(fn,s,xt,row['K'])
        check('sampler_bitwise_replay_'+key,torch.equal(out.cpu(),std))
        print('Evaluation audited',key,flush=True)
    # IndependentNumPy statistics use PDFtwo-gate decision; noise is extra context.
    delta=np.array([p['A']['Dmode']-p['B']['Dmode'] for p in e3['pairs']]);mean=delta.mean();sd=delta.std(ddof=1)
    established=(np.all(delta>0) or np.all(delta<0)) and abs(mean)>=2*sd
    check('E3_independent_paired_statistics',np.allclose(delta,e3['deltas'],atol=1e-15) and abs(mean-e3['mean'])<1e-15 and abs(sd-e3['sample_sd'])<1e-15 and established==(e3['teacher_conclusion']=='различие установлено'))
    check('E3_one_factor',lock['A']['schedule']==lock['B']['schedule']=='linear' and lock['A']['sampler']=='ddpm' and lock['B']['sampler']=='ddim')
    for pair in e3['pairs']:
        check('E3_shared_model_seed_'+str(pair['seed']),pair['A']['source_RUN_ID']==pair['B']['source_RUN_ID'] and pair['A']['initial_seed']==pair['B']['initial_seed'])
        if pair['seed']>1:
            for arm in ['A','B']:
                m=read(Path('runs/lab7')/pair[arm]['RUN_ID']/'manifest.json')
                check('E3_preregistered_evaluation_'+arm+str(pair['seed']),datetime.fromisoformat(m['started_at'])>prereg_time)
    for kind in ['linear','cosine']:
        model,c,_=source_model(idx['training']['E1_'+kind]);model=model.double()
        schedule=make_schedule(c['diffusion']).to(c['runtime']['device'])
        gen=torch.Generator().manual_seed(7901)
        initial=torch.randn(9,2,generator=gen,dtype=torch.float64).to(c['runtime']['device'])
        noise=torch.randn(schedule.T+1,9,2,generator=gen,dtype=torch.float64).to(c['runtime']['device'])
        ddpm,_=ddpm_sample(model,schedule,initial,z=noise)
        ddim,_=ddim_sample(model,schedule,initial,schedule.T,eta=1.,z=noise)
        check('float64_full_DDIMeta1_equals_DDPM_'+kind,float((ddpm-ddim).abs().max())<=1e-6)
        errors=diag[kind]['errors'];check('diagnostic_time_grid_'+kind,len(errors)>=20 and errors[0]['t']==1 and errors[-1]['t']==cfg['diffusion']['T'])
        check('epsilon_x0_identity_'+kind,all(abs(r['E_x0']-r['E_epsilon']/r['SNR'])<=max(1e-6,abs(r['E_x0'])*1e-4) for r in errors))
        check('finite_MC_error_decomposition_'+kind,all(abs(r['E_epsilon']-r['E_epsilon_star']-r['E_net']-r['cross_term'])<1e-5 for r in errors))
        check('exact_Enet_zero_'+kind,all(r['E_net']==0 for r in diag[kind]['exact_errors']))
        trajectories=torch.load(Path('runs/lab7')/diag[kind]['trajectory_RUN_ID']/'trajectories1000.pt',weights_only=False)
        check('1000_full_DDPM_trajectories_'+kind,len(trajectories)==cfg['diffusion']['T']+1 and all(v.shape==(1000,2) for v in trajectories.values()))
        for i,f in enumerate(torch.load(Path('runs/lab7')/diag[kind]['RUN_ID']/'score_fields.pt',weights_only=False)):
            check('weighted_field_'+kind+'_'+str(i),f['points'].shape==(1681,2) and abs(float((f['weights']*f['cosine']).sum())-f['weighted_cosine'])<1e-6)
            norms=f['exact'].norm(dim=1);valid=norms>1e-12
            selfcos=f['exact'][valid].square().sum(1)/norms[valid].square()
            check('exact_field_selfcosine_'+kind+'_'+str(i),torch.allclose(selfcos,torch.ones_like(selfcos),atol=1e-6))
    for r in sup['rare_boundaries']:
        c=base();c['data']['pi0']=r['pi0'];roles,stats=roles_and_stats(c);_,labels=target(c).sample(20000,701,True)
        check('rare_target_retrained_'+str(r['pi0']),int((labels==0).sum())==r['train_component0_count'] and r['normalization']==stats and r['training_source']!=idx['training']['E1_linear'])
        check('rare_raw_role_files_'+str(r['pi0']),all(file_hash(v['path'])==v['sha256'] and torch.equal(torch.load(v['path'],weights_only=False),roles[role]) for role,v in r['data_files'].items()))
    forward_audit=read(R/'forward_regeneration_audit.json')
    check('registered_forward_moments_exactly_reconstructed',forward_audit['status']=='PASS' and file_hash(forward_audit['raw_reconstructed_path'])==forward_audit['sha256'])
    check('Q1_real10T_training',read(Path(sup['Q1']['training_source'])/'manifest.json')['global_step']==20000 and sup['Q1']['quality']['NFE']==10*cfg['diffusion']['T'])
    # PureNumPy variance recursion independent of sampler implementation.
    beta=np.linspace(.0001,.02,cfg['diffusion']['T']);alpha=1-beta;abar=np.r_[1.,np.cumprod(alpha)]
    posterior=beta*(1-abar[:-1])/(1-abar[1:]);variance=1.
    for j in range(len(beta)-1,-1,-1):variance=variance/alpha[j]+(posterior[j] if j>0 else 0.)
    check('zero_predictor_variance_formula',abs(variance/sup['Q2']['theory_variance_standardized']-1)<1e-12)
    # Teacher numerical acceptance outcomes remain separate from evidence integrity.
    for kind in ['linear','cosine']:
        findings.append({'criterion':'E1_forward_moments_'+kind,'PASS':e1['theory'][kind]['all_forward_moments_PASS']})
    findings += [{'criterion':'E2_exact_full_'+r['RUN_ID'],'PASS':r['PASS']} for r in e2['exact_full_acceptance']]
    for pred,v in e2['timing_acceptance'].items():findings.append({'criterion':'E2_time_per_NFE_'+pred,'PASS':all(r['PASS'] for r in v['conditions']),'detail':v})
    import xml.etree.ElementTree as ET
    suite=ET.parse('docs/environment/tests-lab7-final.xml').getroot().find('testsuite')
    check('final_tests',int(suite.attrib['failures'])==int(suite.attrib['errors'])==0)
    answers=read(R/'counterfactual_answers.json');check('five_real_counterfactuals',len(answers)==5 and all(a['RUN_IDs'] for a in answers))
    qa=read(R/'figure_QA.json');provenance=read(R/'figure_provenance.json')
    check('independent_numpy_mode_oracle',read(R/'independent_mode_audit.json')['status']=='PASS')
    check('independent_artifact_regeneration',read(R/'regeneration_audit.json')['status']=='PASS')
    check('figures_visually_verified',qa['status']=='PASS' and len(qa['figures'])==provenance['figure_groups'])
    check('figure_hashes',all(file_hash(R/'figures'/n)==h for n,h in provenance['figure_hashes'].items()))
    for budget in read(R/'budget.json')['measurements']:
        p=Path('runs/lab7')/budget['RUN_ID'];c=yaml.safe_load((p/'config.resolved.yaml').read_text())
        for f in (p/'checkpoints').glob('*.pt'):
            state=torch.load(f,map_location='cpu',weights_only=False);model=build_model(c['model']);model.load_state_dict(state['model'],strict=True)
            check('budget_checkpoint_'+p.name+'_'+f.stem,state['config_hash']==digest(c))
            inventory.append({'path':str(f),'sha256':file_hash(f),'RUN_ID':p.name,'purpose':'budget','global_step':state['global_step']})
    stats={'formal_training_runs':len(idx['training']),'formal_optimizer_updates':20000*len(idx['training']),
        'formal_training_seconds':trainseconds,'formal_checkpoints':sum(r['purpose']=='formal' for r in inventory),'all_checkpoints':len(inventory),
        'budget_training_runs':len(read(R/'budget.json')['measurements']),'evaluation_runs':len(idx['evaluations']),'diagnostic_noise_runs':len(idx['observations']),
        'all_run_folders':len(list(Path('runs/lab7').iterdir())),'tests_passed':int(suite.attrib['tests']),
        'figure_groups':provenance['figure_groups'],'audit_finished_at':datetime.now(timezone.utc).isoformat()}
    first=min(read(Path(p)/'manifest.json')['started_at'] for p in idx['training'].values())
    stats['formal_workflow_wall_seconds']=(datetime.now(timezone.utc)-datetime.fromisoformat(first)).total_seconds()
    begin=ET.parse('docs/environment/tests-lab6-verification-before-lab7.xml').getroot().find('testsuite').attrib['timestamp']
    stats['verification_to_audit_wall_seconds']=(datetime.now(timezone.utc)-datetime.fromisoformat(begin)).total_seconds()
    write_json(R/'checkpoint_inventory.json',inventory)
    raw=list(Path('runs/lab7').glob('*/*.pt'))+list(Path('data/lab7').rglob('*.pt'))+list((R/'metrics').rglob('*.pt'))+[Path(forward_audit['raw_reconstructed_path'])]
    write_json(R/'raw_artifact_inventory.json',[{'path':str(p),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(raw)])
    write_json(R/'execution_statistics.json',stats)
    write_json(R/'final_audit.json',{'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL','checks':checks,'check_count':len(checks),
        'failed':[c for c in checks if c['status']=='FAIL'],'teacher_numeric_findings':findings,'statistics':stats,
        'historical_states':{'Lab4Q2':'BLOCKED','Lab5_course_split':'UNVERIFIED','Lab5_E2_magnitude':'FAIL','Lab6_E1_mixing':'FAIL'},'optimizer_updates_executed_by_audit':0})
    print('FINAL AUDIT',len(checks),'checks; failures:',[c['check'] for c in checks if c['status']=='FAIL'],flush=True)


if __name__=='__main__':main()
