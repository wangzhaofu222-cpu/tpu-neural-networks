"""Validation-only regime lock and append-only test evaluation suite."""
import argparse
import json
import time
from pathlib import Path
from src.engine.runner import ExperimentRunner
from src.utils.io import write_json

ROOT=Path('experiments/lab2')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ledger():return read(ROOT/'ledger.json')
def row(label):return next(r for r in ledger() if r['label']==label)
def evaluate(label,source,corruption='clean',level=0):
    rows=read(ROOT/'evaluations.json') if (ROOT/'evaluations.json').exists() else []
    if any(r['label']==label for r in rows):return
    start=time.perf_counter();path=ExperimentRunner.evaluate_checkpoint(source,corruption,level)
    result=read(path/'summary.json');rows.append({'label':label,'path':path.as_posix(),**result})
    write_json(ROOT/'evaluations.json',rows)
    print(json.dumps({'label':label,'accuracy':result['accuracy'],'RUN_ID':path.name,'seconds':time.perf_counter()-start}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['lock','e1','e3','cf2']);a=p.parse_args()
    from src.config import load_config
    cfg=load_config('configs/lab2.yaml');fractions=cfg['profile']['lab2_fractions']
    if a.stage=='lock':
        if (ROOT/'selection.json').exists():raise RuntimeError('Selection already locked')
        candidates=[]
        for r in fractions:
            runs={arm:row(f'E1_{arm}_r{r:g}_j1') for arm in ['cnn','vit']}
            delta=read(Path(runs['vit']['path'])/'summary.json')['validation_metric']-read(Path(runs['cnn']['path'])/'summary.json')['validation_metric']
            candidates.append({'fraction':r,'validation_delta':delta,'runs':runs})
        selected=max(candidates,key=lambda c:(abs(c['validation_delta']),-c['fraction']))
        write_json(ROOT/'selection.json',{**selected,'all_validation_candidates':candidates,'rule':'max absolute final validation gap, ties smaller r; no test accessed','locked_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
        return
    if a.stage=='e1':
        for r in fractions:
            for arm in ['cnn','vit']:
                label=f'E1_{arm}_r{r:g}_j1';source=row(label)['path'];evaluate(label+'_clean',source)
                for c in cfg['profile']['lab2_corruptions']:
                    for s in [1,2,3]:evaluate(f'{label}_{c}_{s}',source,c,s)
                    if r==1.:evaluate(f'{label}_{c}_0',source,c,0)
                if r==1.:
                    for s in [1,2,3]:evaluate(f'{label}_translation_{s}',source,'translation',s)
    elif a.stage=='e3':
        for arm in ['cnn','vit']:
            for j in [2,3]:
                label=f'E3_{arm}_j{j}';evaluate(label+'_clean',row(label)['path'])
    else:evaluate('CF2_equal_epochs_cnn_clean',row('CF2_equal_epochs_cnn')['path'])

if __name__=='__main__':main()
