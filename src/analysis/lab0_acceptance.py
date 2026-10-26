"""Validate real completed runs. Fixture tests never count as formal experiments."""
import json
from pathlib import Path
import numpy as np
import torch
from src.analysis.statistics import seed_statistics,parameter_relative_difference
from src.config import load_config
from src.compare import diff_configs


def read_run(path):
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='COMPLETED' or manifest.get('synthetic_fixture') or manifest['purpose']!='formal':
        raise ValueError('Acceptance requires completed real formal runs')
    summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
    metrics=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
    steps=[json.loads(line)['train_loss'] for line in (path/'steps.jsonl').read_text().splitlines()]
    return {'path':path,'manifest':manifest,'summary':summary,'metrics':metrics,'steps':steps,
            'config':load_config(path/'config.resolved.yaml')}


def e0(run,evaluation):
    r=read_run(run)
    ev=json.loads(Path(evaluation).read_text())
    if ev['RUN_ID']!=r['manifest']['RUN_ID'] or ev['role']!='val' or ev['checkpoint']!='last.pt':
        raise ValueError('E0 requires independent evaluation of this final checkpoint on val')
    checks={'loss_halved':r['metrics'][-1]['train_loss']<=r['metrics'][0]['train_loss']/2,
            'val_accuracy_30_percent':r['summary']['validation_metric']>=.30,
            'all_epochs_logged':len(r['metrics'])==r['config']['train']['epochs'],
            'independent_evaluation':abs(ev['accuracy']-r['summary']['validation_metric'])<=1e-6}
    return {'checks':checks,'status':'PASS' if all(checks.values()) else 'FAIL'}


def repeatability(a,b,mode='A'):
    a,b=read_run(a),read_run(b)
    if mode not in ['A','B','C']:
        raise ValueError('Unknown reproducibility mode')
    aa,bb=a['manifest'],b['manifest']
    if len(a['steps'])!=len(b['steps']):
        relative_loss=None
    else:
        x,y=np.asarray(a['steps']),np.asarray(b['steps'])
        relative_loss=float(np.max(np.abs(x-y)/np.maximum(1,np.abs(x))))
    acc_diff=abs(a['summary']['validation_metric']-b['summary']['validation_metric'])
    checks={'same_EXP_ID':aa['EXP_ID']==bb['EXP_ID'],'different_RUN_ID':aa['RUN_ID']!=bb['RUN_ID'],
            'same_config':not diff_configs(a['config'],b['config']),
            'same_split':aa['split_fingerprint']==bb['split_fingerprint'],
            'same_first_batch':aa['first_batch_checksum']==bb['first_batch_checksum']}
    if mode!='C':
        checks['loss_trajectory']=relative_loss is not None and relative_loss<=(0 if mode=='A' else 1e-5)
        checks['final_accuracy']=acc_diff<=(0 if mode=='A' else .001)
    return {'mode':mode,'checks':checks,'max_relative_step_loss':relative_loss,'accuracy_difference':acc_diff,
            'status':'PASS' if all(checks.values()) else 'FAIL',
            'mode_C_explanation_required':mode=='C'}


def e2(paths):
    runs=[read_run(p) for p in paths]
    seeds=[r['manifest']['seed'] for r in runs]
    if len(runs)<5 or len(set(seeds))!=len(runs):
        raise ValueError('E2 needs at least five distinct seeds')
    configs=[]
    for r in runs:
        c=json.loads(json.dumps(r['config']))
        c['experiment'].pop('seed')
        configs.append(c)
    if any(diff_configs(configs[0],c) for c in configs):
        raise ValueError('E2 changed more than the seed')
    if len({r['manifest']['split_fingerprint'] for r in runs})!=1:
        raise ValueError('E2 splits differ')
    for key in ['source_hash','git_commit','data_fingerprint']:
        if len({r['manifest'][key] for r in runs})!=1:
            raise ValueError('E2 changed '+key)
    if len({r['manifest']['EXP_ID'] for r in runs})!=len(runs):
        raise ValueError('Different E2 seeds must have different EXP_IDs')
    result=seed_statistics([r['summary']['validation_metric'] for r in runs])
    result.update(RUN_IDs=[r['manifest']['RUN_ID'] for r in runs],large_variance_explanation_required=result['sample_std']>.015)
    return result


def resume_equivalence(a,b,mode='A'):
    a,b=read_run(a),read_run(b)
    aa=torch.load(a['path']/'checkpoints/last.pt',weights_only=False,map_location='cpu')
    bb=torch.load(b['path']/'checkpoints/last.pt',weights_only=False,map_location='cpu')
    delta=parameter_relative_difference(aa['model'],bb['model'])
    acc=abs(a['summary']['validation_metric']-b['summary']['validation_metric'])
    checks={'optimizer_steps':aa['global_step']==bb['global_step']}
    if mode!='C':
        checks.update(parameter_difference=delta<=(1e-7 if mode=='A' else 1e-5),accuracy=acc<=(0 if mode=='A' else .001))
    return {'delta_theta':delta,'accuracy_difference':acc,'checks':checks,'status':'PASS' if all(checks.values()) else 'FAIL'}
