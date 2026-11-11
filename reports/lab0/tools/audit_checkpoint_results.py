"""Read-only arithmetic verification of the saved checkpoints; no training."""
from pathlib import Path
import json
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/lab0'
rows=json.loads((OUT/'run_registry.json').read_text('utf-8'))
runs={r['condition']:ROOT/'runs'/r['RUN_ID'] for r in rows}
cache={}
def state(key):
    if key not in cache:
        cache[key]=torch.load(runs[key]/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    return cache[key]
def delta(a,b):
    a,b=state(a)['model'],state(b)['model']
    assert a.keys()==b.keys()
    aa=torch.cat([x.reshape(-1).double() for x in a.values()])
    bb=torch.cat([x.reshape(-1).double() for x in b.values()])
    return float(torch.linalg.vector_norm(aa-bb)/(torch.linalg.vector_norm(aa)+1e-12))

checks={}
for mode in 'ABC':
    d=delta(f'E1-{mode}-1',f'E1-{mode}-2');assert d==0
    checks[f'E1-{mode}-parameter-delta']=d
    d=delta(f'E3-{mode}-continuous',f'E3-{mode}-resumed');assert d==0
    checks[f'E3-{mode}-parameter-delta']=d
    for suffix in ['continuous','resumed']:
        s=state(f'E3-{mode}-{suffix}')
        assert s['epoch']==12 and s['global_step']==(2015 if mode=='C' else 2016)
d=delta('E3-A-continuous','CF03-weights-only')
assert abs(d-.3710280114273825)<1e-7
checks['CF03-double-precision-delta']=d
checks['parameter_count']=sum(x.numel() for x in state('E1-A-1')['model'].values())
assert checks['parameter_count']==252778
a,b=[runs[k]/'steps.jsonl' for k in ['CF04-extra-1','CF04-extra-3']]
aa=[json.loads(x)['train_loss'] for x in a.read_text('utf-8').splitlines()]
bb=[json.loads(x)['train_loss'] for x in b.read_text('utf-8').splitlines()]
checks['CF04-differing-step-losses']=sum(x!=y for x,y in zip(aa,bb))
checks['CF04-max-relative-loss']=max(abs(x-y)/max(1,abs(x)) for x,y in zip(aa,bb))
assert len(aa)==len(bb)==checks['CF04-differing-step-losses']==2016
assert abs(checks['CF04-max-relative-loss']-.5109808444976807)<1e-12
e=json.loads((ROOT/'docs/environment/e0-selected.json').read_text('utf-8'))
ev=json.loads((ROOT/e['evaluation_path']).read_text('utf-8'))
checks['independent_evaluation']=ev
checks['checkpoints_loaded']=len(cache)
checks['training_executed']=False
(OUT/'build/checkpoint_numeric_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),'utf-8')
print('Read-only checkpoint arithmetic verified:',len(cache),'checkpoints.')
