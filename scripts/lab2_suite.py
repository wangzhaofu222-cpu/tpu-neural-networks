"""Lab2 orchestration; optimization is exclusively the shared ExperimentRunner."""
import argparse,copy,json,math,time
from pathlib import Path
from src.config import load_config,validate
from src.engine.runner import ExperimentRunner
from src.utils.io import write_json

ROOT=Path('experiments/lab2')
MODELS={'cnn':'cifar_resnet18','vit':'cifar_vit'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def records():return read(ROOT/'ledger.json') if (ROOT/'ledger.json').exists() else []

def add_record(label,path,elapsed,kind='training',**extra):
    rows=records();m=read(path/'manifest.json')
    row={'label':label,'RUN_ID':path.name,'path':path.as_posix(),'kind':kind,'status':m['status'],
         'elapsed_seconds':elapsed,**extra};rows.append(row);write_json(ROOT/'ledger.json',rows)
    print(json.dumps({'finished':label,**row}),flush=True);return path

def execute(label,cfg,profile=False):
    found=next((r for r in records() if r['label']==label),None)
    if found:return Path(found['path'])
    cfg=copy.deepcopy(cfg);cfg['experiment']['name']=label;validate(cfg)
    start=time.perf_counter()
    try:
        runner=ExperimentRunner(cfg);path=runner.run_profile() if profile else runner.run()
    except FloatingPointError as exc:
        if not getattr(exc,'run_path',None):raise
        path=Path(exc.run_path)
    return add_record(label,path,time.perf_counter()-start,'profiling' if profile else 'training')

def alias(label,source):
    found=next((r for r in records() if r['label']==label),None)
    if found:return Path(found['path'])
    return add_record(label,Path(source),0.,kind='alias',reuse_reason='Identical completed scientific condition, not a new training execution')

def base_config():
    cfg=load_config('configs/lab2.yaml');cfg['experiment']['purpose']='formal'
    cfg['logging']['profile_on_finish']=True
    return cfg

def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','budget','lr','e1','e3','cf2','profile2'])
    p.add_argument('--suite-root',default='experiments/lab2');a=p.parse_args();ROOT=Path(a.suite_root);ROOT.mkdir(parents=True,exist_ok=True)
    base=base_config()
    if a.stage in {'smoke','budget'}:
        base['experiment'].update(purpose=a.stage,role=a.stage)
        base['data']['train_fraction']=.01 if a.stage=='smoke' else 1.
        base['train']['max_steps']=4 if a.stage=='smoke' else 336
        base['logging']['profile_on_finish']=a.stage=='budget'
        for arm,name in MODELS.items():
            base['model']['name']=name;execute(f'{a.stage}_{arm}',base)
        if a.stage=='budget':
            per={arm:read(Path(next(r['path'] for r in records() if r['label']==f'budget_{arm}'))/'summary.json')['training_seconds']/336 for arm in MODELS}
            write_json(ROOT/'budget.json',{'seconds_per_update_including_one_final_validation_and_checkpoint':per,
                'primary_updates':180000,'CF2_updates':506,'estimated_training_seconds':90000*sum(per.values())+506*per['cnn'],
                'formal_training_executions':19,'architecture':'canonical GPU pair; float32; batch128; unchanged 10000-step primary budget'})
        return
    if a.stage=='lr':
        for arm,name in MODELS.items():
            base['model']['name']=name
            for lr in [.0003,.001,.003]:
                base['optimizer']['lr']=lr;execute(f'LR_{arm}_{lr}_j1',base)
        selected={}
        for arm in MODELS:
            candidates=[]
            for lr in [.0003,.001,.003]:
                row=next(r for r in records() if r['label']==f'LR_{arm}_{lr}_j1');s=read(Path(row['path'])/'summary.json')
                if row['status']=='COMPLETED':candidates.append((s['validation_metric'],-s['validation_loss'],-lr,row['path']))
            best=max(candidates);selected[arm]={'lr':-best[2],'path':best[3],'RUN_ID':Path(best[3]).name,'validation_accuracy':best[0]}
        write_json(ROOT/'selected_lr.json',selected);return
    selected=read(ROOT/'selected_lr.json')
    if a.stage=='e1':
        for arm,name in MODELS.items():
            base['model']['name']=name;base['optimizer']['lr']=selected[arm]['lr']
            for fraction in base['profile']['lab2_fractions']:
                label=f'E1_{arm}_r{fraction:g}_j1'
                if fraction==1.:alias(label,selected[arm]['path'])
                else:
                    base['data']['train_fraction']=fraction;execute(label,base)
    elif a.stage=='e3':
        pre=read(ROOT/'selection.json')
        import subprocess
        hyp_path=ROOT/'hypothesis.md'
        committed=subprocess.run(['git','-c','safe.directory=D:/神经网络','show','HEAD:'+hyp_path.as_posix()],capture_output=True,text=True,encoding='utf-8',check=True).stdout
        if committed!=hyp_path.read_text(encoding='utf-8'):raise RuntimeError('Final Lab2 hypothesis is not committed')
        for arm,name in MODELS.items():
            base['model']['name']=name;base['optimizer']['lr']=selected[arm]['lr'];base['data']['train_fraction']=pre['fraction']
            first=next(r for r in records() if r['label']==f"E1_{arm}_r{pre['fraction']:g}_j1")
            alias(f'E3_{arm}_j1',first['path'])
            for seed in [2,3]:
                base['experiment']['seed']=seed;execute(f'E3_{arm}_j{seed}',base)
    elif a.stage=='cf2':
        base['model']['name']=MODELS['cnn'];base['optimizer']['lr']=selected['cnn']['lr']
        base['data']['train_fraction']=base['profile']['lab2_fractions'][0]
        base['train']['max_steps']=math.ceil((10000/math.ceil(43000/128))*math.ceil(int(43000*base['data']['train_fraction'])/128))
        base['experiment']['role']='CF2';execute('CF2_equal_epochs_cnn',base)
    else:
        base['model'].update(name=MODELS['vit'],patch_size=2)
        base['experiment']['role']='profiling';execute('CF4_ViT_patch2_profile',base,profile=True)

if __name__=='__main__':main()
