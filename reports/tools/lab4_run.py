"""Serial, restart-aware Lab4 orchestration. Training always uses ExperimentRunner."""
import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import torch
import yaml
from src.config import load_config,validate
from src.engine.runner import ExperimentRunner
from src.evaluation.ssl_evaluation import feature_cache,evaluate_probe
from src.utils.io import write_json,file_hash

ROOT=Path('experiments/lab4');CONTROL=ROOT/'control'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def base():return load_config('configs/lab4.yaml')
def rows(name):return read(ROOT/name) if (ROOT/name).exists() else []
def append(name,row):
    values=rows(name);values.append(row);write_json(ROOT/name,values)
def lookup(label,name='ledger.json'):
    return next((v for v in reversed(rows(name)) if v['label']==label and v['status'].startswith('COMPLETED')),None)
def hardware():
    q=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    return {'at_utc':datetime.now(timezone.utc).isoformat(),'nvidia_smi':q.stdout.strip(),'returncode':q.returncode}
def dispatch(args):return subprocess.run([sys.executable,'-X','utf8','-m','reports.tools.lab4_run',*args],check=False)

def train(label,cfg,initialize=False,allow_failure=False):
    found=lookup(label)
    if found:return Path(found['path'])
    CONTROL.mkdir(exist_ok=True);config=CONTROL/f'{label}.yaml'
    config.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8');validate(cfg)
    start=time.perf_counter();before=hardware()
    cmd=['worker_train','--config',str(config),'--label',label]
    if initialize:cmd.append('--initialize')
    result=dispatch(cmd);out=CONTROL/f'{label}.json'
    if not out.exists():raise RuntimeError(f'Training worker interrupted; inspect checkpoint/run status for{label}')
    record=read(out);record.update(label=label,kind='initialization' if initialize else 'training',
        purpose=cfg['experiment']['purpose'],elapsed_seconds=time.perf_counter()-start,hardware_before=before,hardware_after=hardware())
    append('ledger.json',record)
    if result.returncode or not record['status'].startswith('COMPLETED'):
        if allow_failure:return None
        raise RuntimeError(f'Training failed:{label}; validcheckpoint retained in{record.get("path")}')
    return Path(record['path'])

def worker_train(config,label,initialize=False):
    cfg=load_config(config)
    try:
        p=ExperimentRunner(cfg).run(initialize_only=initialize)
        record={'path':p.as_posix(),'RUN_ID':p.name,**read(p/'summary.json')}
        write_json(CONTROL/f'{label}.json',record)
    except Exception as exc:
        p=Path(exc.run_path) if hasattr(exc,'run_path') else None
        write_json(CONTROL/f'{label}.json',{'status':'FAILED','path':p.as_posix() if p else None,'RUN_ID':p.name if p else None,
                   'exception':type(exc).__name__+':'+str(exc)})
        raise

def features(label,source,checkpoint='last.pt'):
    old=lookup(label,'features.json')
    if old:return Path(old['path'])
    result=dispatch(['worker_features','--label',label,'--source',str(source),'--checkpoint',checkpoint])
    if result.returncode:raise RuntimeError('Feature extraction failed:'+label)
    return Path(lookup(label,'features.json')['path'])

def probe(label,feature_run,seed=1,fraction=1.,kind='h',budget=False):
    old=lookup(label,'evaluations.json')
    if old:return old
    args=['worker_probe','--label',label,'--source',str(feature_run),'--seed',str(seed),'--fraction',str(fraction),'--kind',kind]
    if budget:args.append('--budget')
    if dispatch(args).returncode:raise RuntimeError('Probe evaluation failed:'+label)
    return lookup(label,'evaluations.json')

def smoke():
    write_json(ROOT/'hardware_initial.json',hardware())
    for amp,label in [(False,'memory_N256_float32'),(True,'smoke_N256_AMP')]:
        c=base();c['experiment']['purpose']='smoke';c['train'].update(amp=amp,max_steps=1);c['representation']['enabled']=False
        c['data']['num_workers']=0;train(label,c,allow_failure=True)
    if not lookup('smoke_N256_AMP'):raise RuntimeError('N256 does not fit evenwithAMP; user decision required, no protocol downsizing')
    c=base();c['experiment']['purpose']='budget';c['train']['epochs']=1;c['logging']['keep_epochs']=[1]
    p=train('budget_N256_epoch1',c);s=read(p/'summary.json');m=read(p/'manifest.json')
    write_json(ROOT/'budget.json',{'RUN_ID':p.name,'actual_N':256,'negative_count':510,'AMP':True,
        'simultaneous_views':512,'NT_Xent_denominator_candidates':511,'loss_dtype':'float32',
        'training_peak_allocator_bytes':s['peak_gpu_memory_bytes'],'measured_training_seconds_including_epoch0diagnostics':s['training_seconds'],
        'measured_epoch_seconds':s['epoch_time'],'estimated_Efull100_seconds':s['epoch_time']*100,
        'estimated_Eshort25_seconds':s['epoch_time']*25,'estimated325epochs_conservative_seconds':s['epoch_time']*325,
        'measured_steps':s['global_step'],'images_per_epoch':42752,'images_per_second_including_diagnostics':42752/s['epoch_time'],
        'model':m['environment']['device_name'],'scope':'Budgetisnotformal; no CPU reducedprotocol andno microbatch independent losses.'})
    f=features('budget_features',p);probe('budget_probe',f,budget=True)

def e1():
    c=base();c['experiment'].update(purpose='formal',role='B0');b0=train('E1_B0',c,initialize=True)
    c=base();c['experiment'].update(purpose='formal',role='B1');c['model']['mode']='supervised';c['train']['epochs']=25
    c['optimizer'].update(lr=.1,weight_decay=.0005);c['augmentations']=[c['augmentations'][0],c['augmentations'][2]]
    c['representation']['interval']=25;c['logging']['keep_epochs']=[25];b1=train('E1_B1',c)
    c=base();c['experiment'].update(purpose='formal',role='E1_SSL');ssl=train('E1_SSL',c)
    for name,p in [('B0',b0),('B1',b1)]:
        f=features('E1_'+name+'_features',p);probe('E1_'+name+'_h',f)
    checkpoints={}
    for epoch in [0,25,50,75,100]:
        f=features(f'E1_SSL_epoch{epoch}_features',ssl,f'epoch_{epoch:03}.pt')
        evaluation=probe(f'E1_SSL_epoch{epoch}_h',f);checkpoints[str(epoch)]={'features':f.as_posix(),'evaluation_RUN_ID':evaluation['RUN_ID']}
    write_json(ROOT/'E1_sources.json',{'B0':b0.as_posix(),'B1':b1.as_posix(),'SimCLR':ssl.as_posix(),'SSL_checkpoints':checkpoints})

def e2():
    full=base()['augmentations'];variants={'T':full,**{'minus_'+v['name']:[o for o in full if o['name']!=v['name']] for v in full}}
    evaluations={}
    for name,ops in variants.items():
        c=base();c['experiment'].update(purpose='formal',role='E2');c['train']['epochs']=25;c['augmentations']=ops
        c['representation']['interval']=25;c['logging']['keep_epochs']=[25]
        p=train('E2_'+name,c);f=features('E2_'+name+'_features',p);evaluations[name]=probe('E2_'+name+'_h',f)
    # Globalprotocol requires allselectionbyval. Testdeltas are reported separately after locking this operation.
    candidates=[{'operation':op['name'],'validation_delta':evaluations['T']['validation_accuracy']-evaluations['minus_'+op['name']]['validation_accuracy']} for op in full]
    chosen=max(candidates,key=lambda r:r['validation_delta'])['operation']
    write_json(ROOT/'E2_single_selection.json',{'operation':chosen,'candidates':candidates,'criterion':'largest valprobe delta, ties baseconfigorder; testnotusedtoselectaugmentation'})
    for name,ops in [('only_'+chosen,[o for o in full if o['name']==chosen]),('none',[])]:
        c=base();c['experiment'].update(purpose='formal',role='E2');c['train']['epochs']=25;c['augmentations']=ops
        c['representation']['interval']=25;c['logging']['keep_epochs']=[25]
        p=train('E2_'+name,c);f=features('E2_'+name+'_features',p);probe('E2_'+name+'_h',f)

def e3():
    proof=read(ROOT/'preregistration_evidence.json');blob=subprocess.run(['git','-c','safe.directory=D:/神经网络','show',proof['git_commit']+':experiments/lab4/hypothesis.md'],capture_output=True,text=True,encoding='utf-8',check=True).stdout
    assert blob==(ROOT/'hypothesis.md').read_text(encoding='utf-8')
    for seed in [1,2,3]:
        for fraction in base()['profile']['lab4_label_fractions']:
            for name,label in [('B0','E1_B0_features'),('B1','E1_B1_features'),('SimCLR','E1_SSL_epoch100_features')]:
                f=Path(lookup(label,'features.json')['path'])
                if seed==1 and fraction==1.:
                    old=lookup('E1_'+name+'_h' if name!='SimCLR' else 'E1_SSL_epoch100_h','evaluations.json')
                    write_json(ROOT/f'E3_reuse_{name}.json',{'reused_E1_evaluation_RUN_ID':old['RUN_ID'],'reason':'identical seed1 fulltrain probe; no unnecessary retraining'})
                    continue
                probe(f'E3_{name}_s{seed}_r{fraction:g}',f,seed,fraction)

def counterfactuals():
    c=base();c['experiment'].update(purpose='formal',role='CF1');c['train'].update(epochs=25,batch_size=64)
    c['representation']['interval']=25;c['logging']['keep_epochs']=[25];train('CF1_N64',c)
    f=Path(lookup('E1_SSL_epoch100_features','features.json')['path']);probe('CF4_SSL_z_hat',f,kind='z_hat')
    c=base();c['experiment'].update(purpose='formal',role='CF5',subset_seed=1);c['model']['mode']='supervised';c['train']['epochs']=25
    c['data']['train_fraction']=min(c['profile']['lab4_label_fractions']);c['optimizer'].update(lr=.1,weight_decay=.0005)
    c['augmentations']=[c['augmentations'][0],c['augmentations'][2]];c['representation']['interval']=25;c['logging']['keep_epochs']=[25]
    p=train('CF5_B1_low_labels',c);f=features('CF5_B1_low_labels_features',p)
    probe('CF5_B1_low_labels_h',f,seed=1,fraction=min(c['profile']['lab4_label_fractions']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','e1','e2','e3','counterfactuals','worker_train','worker_features','worker_probe'])
    for name in ['config','label','source']:parser.add_argument('--'+name)
    parser.add_argument('--checkpoint',default='last.pt');parser.add_argument('--initialize',action='store_true');parser.add_argument('--budget',action='store_true')
    parser.add_argument('--seed',type=int,default=1);parser.add_argument('--fraction',type=float,default=1.);parser.add_argument('--kind',default='h');a=parser.parse_args()
    if a.stage=='worker_train':worker_train(a.config,a.label,a.initialize)
    elif a.stage=='worker_features':
        p,r=feature_cache(a.source,a.checkpoint);append('features.json',{'label':a.label,'path':p.as_posix(),**r});print('FEATURES',a.label,p,flush=True)
    elif a.stage=='worker_probe':
        c=base();c['experiment']['purpose']='budget' if a.budget else 'formal'
        if a.budget:c['evaluation'].update(probe_epochs=2,regularization_grid=[.001])
        p,r=evaluate_probe(c,a.source,a.seed,a.fraction,a.kind,a.label);append('evaluations.json',{'path':p.as_posix(),**r})
        print('PROBE',a.label,r['test_accuracy'],r['knn_test_accuracy'],flush=True)
    else:globals()[a.stage]()
