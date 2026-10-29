"""Restartable serial stages. All optimization dispatches to shared ExperimentRunner."""
import argparse,copy,json,math
from pathlib import Path
from datetime import datetime,timezone
import torch
from src.generative.experiments import config,train,evaluate,noise_baseline,quadratic_evaluation,read
from src.diagnostics.density_metrics import select_flow_depth,paired_statistics
from src.analysis.generative_comparison import match_parameters
from src.utils.io import write_json,digest
from src.utils.preregistration import verify_git_hypothesis

R=Path('experiments/lab6');INDEX=R/'run_index.json'
def index():return read(INDEX) if INDEX.exists() else {'training':{},'evaluations':{},'quadratic':{}}
def train_once(key,cfg):
    idx=index()
    if key not in idx['training']:
        p=train(cfg);s=read(Path(p)/'summary.json')
        if s['status']!='COMPLETED' or s['global_step']!=5000:raise RuntimeError('Formal training paused/failed: '+p)
        idx['training'][key]=p;write_json(INDEX,idx)
    return idx['training'][key]
def evaluate_once(key,source,**kwargs):
    idx=index()
    if key not in idx['evaluations']:
        idx['evaluations'][key]=evaluate(source,**kwargs);write_json(INDEX,idx)
    return idx['evaluations'][key]

def e1():
    noise_baseline('two_blobs')
    source=train_once('E1_EBM',config(role='E1_EBM'))
    etas=read(R/'profile.json')['profile']['lab6_eta'];rows=[];quadratic=[]
    for i,eta in enumerate(etas):
        r=evaluate_once(f'E1_eta_{eta}',source,eta=eta,center_probe=True)
        rows.append({'eta':eta,'MMD2':r['MMD2'],'Cov':r['Cov'],'Dmode':r['Dmode'],'DK':r['sampler']['DK'],
            'outside_final_fraction':r['outside_final_fraction'],'pre_reflection_outside_proposal_fraction':r['sampler']['pre_reflection_outside_proposal_fraction'],
            'stable':r['stable'],'center_to_other_mode_fraction':r['center_start_mixing']['final_other_mode_fraction'],
            'RUN_ID':r['RUN_ID'],'source_RUN_ID':r['source_RUN_ID']})
        idx=index();key=f'eta_{eta}'
        if key not in idx['quadratic']:
            idx['quadratic'][key]=quadratic_evaluation(eta,8101+i);write_json(INDEX,idx)
        quadratic.append(idx['quadratic'][key])
    idx=index()
    if 'wrong_noise_control' not in idx['quadratic']:
        idx['quadratic']['wrong_noise_control']=quadratic_evaluation(max(etas),8105,etas[1]);write_json(INDEX,idx)
    candidates=[r for r in rows if r['stable'] and r['center_to_other_mode_fraction']>=.2]
    selected=min(candidates,key=lambda r:r['eta']) if candidates else min([r for r in rows if r['stable']],key=lambda r:r['MMD2'])
    detail=index()['evaluations'][f"E1_eta_{selected['eta']}"]
    result={'source':source,'eta_rows':rows,'quadratic':quadratic,'wrong_noise_control':idx['quadratic']['wrong_noise_control'],
        'selected_eta':selected['eta'],'mixing_rule_satisfied':bool(candidates),
        'selection_rule':'minimum eta stable with center endpoint mixing>=20%' if candidates else 'fallback stable minimumMMD2; K50 mixing criterion not met',
        'center_energies':detail['center_energies'],'origin_energy':detail['origin_energy'],'mean_test_energy':detail['mean_test_energy'],
        'tail_bound':detail['tail_unnormalized_integral_upper_bound_R14'],
        'center_energy_acceptance':'PASS' if max(detail['center_energies'])<detail['origin_energy'] else 'FAIL',
        'variance_acceptance':'PASS' if all(r['relative_error']<=.05 for r in quadratic+[idx['quadratic']['wrong_noise_control']]) else 'FAIL',
        'too_small_eta':min(etas),'too_large_eta_by_theoretical_discretization':max(etas),
        'source_checkpoint_sha256':detail['checkpoint_sha256'],'selected_at':datetime.now(timezone.utc).isoformat()}
    write_json(R/'E1.json',result);print('E1 COMPLETE',json.dumps({'selected_eta':result['selected_eta'],'mixing':result['mixing_rule_satisfied'],'center_acceptance':result['center_energy_acceptance'],'variance_acceptance':result['variance_acceptance']}),flush=True)

def e2():
    rows=[];nll={}
    for depth in [2,4,8,16]:
        source=train_once(f'E2_Flow_{depth}',config(kind='flow',depth=depth,role='E2_Flow'))
        r=evaluate_once(f'E2_Flow_{depth}',source)
        rows.append({'depth':depth,**{k:r[k] for k in ['val_NLL','val_NLL_SEM','MMD2','Cov','Dmode','parameters','sample_seconds','exact_test_NLL','RUN_ID','source_RUN_ID','invertibility']}})
        nll[depth]=torch.load(Path('runs/lab6')/r['RUN_ID']/'val_pointwise_nll.pt',weights_only=False)
    selection=select_flow_depth(rows,nll)
    result={'rows':rows,'selection':selection,'selected_at':datetime.now(timezone.utc).isoformat()}
    write_json(R/'E2.json',result)
    match=match_parameters(selection['selected_depth'],4.);write_json(R/'E3_capacity_match.json',match)
    print('E2 COMPLETE',json.dumps({'selected_depth':selection['selected_depth'],'capacity_match':match}),flush=True)

def e3():
    prereg=read(R/'preregistration_evidence.json');verify_git_hypothesis(prereg['commit'],'experiments/lab6/hypothesis.md')
    lock=read(R/'E3_protocol_lock.json');match=read(R/'E3_capacity_match.json');e1=read(R/'E1.json');e2=read(R/'E2.json')
    if lock['eta']!=e1['selected_eta'] or lock['depth']!=e2['selection']['selected_depth'] or lock['capacity_match']!=match:raise ValueError('Preregistered selection changed')
    if match['relative_gap_over_smaller']>.2:raise ValueError('Capacity match failed')
    target=read(R/'profile.json')['profile']['lab6_target'];noise=noise_baseline(target);rows=[]
    for seed in [1,2,3]:
        pair={}
        for kind in ['ebm','flow']:
            c=config(kind=kind,target=target,seed=seed,role='E3_'+kind,width=match['ebm_width'] if kind=='ebm' else 64,depth=lock['depth'],eta=lock['eta'])
            if digest(c)!=lock['config_hashes'][f'{kind}_{seed}']:raise ValueError('E3 config differs from preregistered resolved configuration')
            source=train_once(f'E3_{kind}_{seed}',c);pair[kind]=evaluate_once(f'E3_{kind}_{seed}',source,seed=7000+seed)
        rows.append({'seed':seed,**pair})
    metric=lock['primary_metric'];noise_value=noise['noise_'+metric]
    paired=paired_statistics([r['ebm'][metric] for r in rows],[r['flow'][metric] for r in rows],noise_value)
    result={'target':target,'paired_runs':rows,'capacity_match':match,'primary_metric':metric,'difference_direction':'EBM minus Flow',
        'paired_statistics':paired,'preregistered_expected_sign':'positive','hypothesis_supported':paired['mean']>0 and paired['conclusion']=='различие установлено',
        'noise':noise,'preregistration_commit':prereg['commit']}
    write_json(R/'E3.json',result);print('E3 COMPLETE',json.dumps(paired,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['e1','e2','e3']);a=p.parse_args();globals()[a.stage]()
