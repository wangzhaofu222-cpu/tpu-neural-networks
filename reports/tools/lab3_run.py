"""Serial Lab3 orchestration; every training operation uses ExperimentRunner."""
import argparse
import copy
import json
import subprocess
import time
import sys
from datetime import datetime,timezone
from pathlib import Path
import torch
import yaml
from src.config import load_config,validate
from src.engine.runner import ExperimentRunner
from src.diagnostics.extrapolation import select_learning_rate
from src.utils.io import write_json,file_hash

ROOT=Path('experiments/lab3');LEDGER=ROOT/'ledger.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ledger():return read(LEDGER) if LEDGER.exists() else []
def record(row):
    values=ledger();values.append(row);write_json(LEDGER,values)
def lookup(label):return next((r for r in ledger() if r['label']==label and r['status'].startswith('COMPLETED')),None)
def hardware():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    return {'at_utc':datetime.now(timezone.utc).isoformat(),'nvidia_smi':p.stdout.strip(),'returncode':p.returncode}
def base():return load_config('configs/lab3.yaml')
def train(label,cfg,resume=None):
    old=lookup(label)
    if old:return Path(old['path'])
    validate(cfg);before=hardware();start=time.perf_counter()
    try:p=ExperimentRunner(cfg).run(resume=resume)
    except Exception as exc:
        if hasattr(exc,'run_path'):
            p=Path(exc.run_path);record({'label':label,'kind':'training','path':p.as_posix(),'RUN_ID':p.name,'status':'FAILED',
                'purpose':cfg['experiment']['purpose'],'exception':str(exc),'seconds':time.perf_counter()-start})
        raise
    s=read(p/'summary.json')
    record({'label':label,'kind':'training','path':p.as_posix(),'RUN_ID':p.name,'status':s['status'],
            'purpose':cfg['experiment']['purpose'],'steps':s['global_step'],'seconds':time.perf_counter()-start,
            'hardware_before':before,'hardware_after':hardware(),'resumed_from':str(resume) if resume else None})
    print('FINISHED',label,p,s['validation_metric'],flush=True);return p
def evaluate(label,source,length,rule='P2',diagnostics=False):
    old=lookup(label)
    if old:return read(Path(old['path'])/'summary.json')
    p,r=ExperimentRunner.evaluate_sequence(source,length,rule,diagnostics=diagnostics)
    record({'label':label,'kind':'attention' if diagnostics else 'evaluation','path':p.as_posix(),'RUN_ID':p.name,'status':r['status'],**r})
    print('EVALUATED',label,r.get('accuracy'),flush=True);return r

def smoke():
    c=base();c['experiment']['purpose']='smoke';c['train']['max_steps']=2
    p=train('smoke',c)
    c['experiment']['purpose']='budget';c['train']['max_steps']=100
    p=train('budget',c);s=read(p/'summary.json')
    write_json(ROOT/'budget.json',{'hardware':hardware(),'RUN_ID':p.name,'measured100_steps_seconds':s['training_seconds'],
        'training_peak_gpu_memory_bytes':s['peak_gpu_memory_bytes'],
        'seconds_per_step_including_validation_checkpoint':s['training_seconds']/100,
        'estimated_2000_step_run_seconds':s['training_seconds']*20,
        'maximum_planned_scientific_updates':10*8000,
        'scope':'Smoke and budget are not scientific formal experiments; conservative estimate includes validation and checkpoint overhead.'})

def lr():
    rows=[]
    for value in [.0003,.001,.003]:
        c=base();c['experiment'].update(purpose='formal',role='LR');c['optimizer']['lr']=value
        p=train(f'LR_{value:g}',c);s=read(p/'summary.json')
        rows.append({'lr':value,'role':'val','length':c['data']['L0'],'position':'sinusoidal','accuracy':s['validation_metric'],'path':p.as_posix(),'RUN_ID':p.name})
    winner=select_learning_rate(rows,base()['data']['L0'])
    write_json(ROOT/'selected_lr.json',{'winner':winner,'candidates':rows,'criterion':'max final valL0 accuracy, ties smallerLR; no test read'})

def e1():
    winner=read(ROOT/'selected_lr.json')['winner'];paths={};steps=2000
    for pe in ['none','absolute','sinusoidal','rope']:
        if pe=='sinusoidal':p=Path(winner['path'])
        else:
            c=base();c['experiment']['purpose']='formal';c['model']['position']=pe;c['optimizer']['lr']=winner['lr']
            p=train(f'E1_{pe}_S2000',c)
        paths[pe]=p.as_posix()
    progression=[]
    while True:
        values={pe:read(Path(p)/'summary.json')['validation_metric'] for pe,p in paths.items()}
        gap=max(v for pe,v in values.items() if pe!='none')-values['none']
        progression.append({'steps':steps,'validation':values,'gap':gap,'paths':copy.deepcopy(paths)})
        write_json(ROOT/'E1_training.json',{'steps':steps,'paths':paths,'progression':progression,'common_lr':winner['lr']})
        if gap>=.20 or steps>=8000:break
        steps*=2
        for pe in paths:
            parent=Path(paths[pe]);c=yaml.safe_load((parent/'config.resolved.yaml').read_text(encoding='utf-8'));c['train']['max_steps']=steps
            paths[pe]=train(f'E1_{pe}_S{steps}',c,parent/'checkpoints/last.pt').as_posix()
    for pe,p in paths.items():
        for length in [128,256,512,1024]:evaluate(f'E1_{pe}_L{length}',p,length)
    evaluate('E1_absolute_P1_L256',paths['absolute'],256,'P1')

def e2():
    for backend in ['explicit','efficient']:
        for length in [64,128,256,512,1024]:
            label=f'E2_isolated_{backend}_L{length}'
            if lookup(label):continue
            subprocess.run([sys.executable,'-X','utf8','-m','reports.tools.lab3_run','one_profile','--backend',backend,'--length',str(length)],check=True)

def one_profile(backend,length,width=128,cf5=False):
    label=f'CF5_isolated_d{width}_L{length}' if cf5 else f'E2_isolated_{backend}_L{length}'
    if lookup(label):return
    c=base();c['experiment'].update(purpose='formal',role='CF5' if cf5 else 'profiling');c['model'].update(position='none',backend=backend,width=width,ffn=width*2)
    c['profiling'].update(dtype='float32' if cf5 else 'float16',length=length,process_isolation=True)
    p,r=ExperimentRunner(c).run_sequence_profile(length,torch.float32 if cf5 else torch.float16)
    record({'label':label,'kind':'profiling','path':p.as_posix(),**r})
    print('PROFILE',label,{k:r.get(k) for k in ['RUN_ID','status','forward_seconds','training_step_seconds','peak_memory_bytes']},flush=True)

def e3():
    selection=read(ROOT/'selection.json');prereg=read(ROOT/'preregistration_evidence.json')
    current=subprocess.run(['git','-c','safe.directory=D:/神经网络','show',prereg['git_commit']+':experiments/lab3/hypothesis.md'],capture_output=True,check=True).stdout
    assert current== (ROOT/'hypothesis.md').read_bytes(), 'Hypothesis changed after preregistration'
    e1data=read(ROOT/'E1_training.json')
    for j in [1,2,3]:
        for pe in selection['positions']:
            if j==1:p=Path(e1data['paths'][pe])
            else:
                c=base();c['experiment'].update(purpose='formal',role='E3',seed=j);c['model']['position']=pe
                c['train']['max_steps']=e1data['steps'];c['optimizer']['lr']=e1data['common_lr'];p=train(f'E3_{pe}_j{j}',c)
            evaluate(f'E3_{pe}_j{j}_L{selection["L_star"]}',p,selection['L_star'])

def diagnostics():
    e=read(ROOT/'E1_training.json');selection=read(ROOT/'selection.json');pe=selection['positions'][0]
    for length in [128,256]:evaluate(f'attention_{pe}_L{length}',e['paths'][pe],length,diagnostics=True)
    for width in [128,256]:
        label=f'CF5_isolated_d{width}_L512'
        if lookup(label):continue
        subprocess.run([sys.executable,'-X','utf8','-m','reports.tools.lab3_run','one_profile','--backend','explicit','--length','512','--width',str(width),'--cf5'],check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','lr','e1','e2','e3','diagnostics','one_profile'])
    parser.add_argument('--backend');parser.add_argument('--length',type=int);parser.add_argument('--width',type=int,default=128);parser.add_argument('--cf5',action='store_true')
    args=parser.parse_args()
    if args.stage=='one_profile':one_profile(args.backend,args.length,args.width,args.cf5)
    else:globals()[args.stage]()
