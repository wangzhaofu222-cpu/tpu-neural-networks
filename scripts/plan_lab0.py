"""Generate a nonexecuting formal run ledger after measuring a real epoch."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.utils.io import write_json


def main():
    budget_path=ROOT/'docs/environment/lab0_budget.json'
    if not budget_path.exists():
        raise SystemExit('Measure one real CPU epoch first; no budget has been established.')
    budget=json.loads(budget_path.read_text())
    cfg=load_config(ROOT/'configs/lab0.yaml')
    seed=cfg['profile']['base_seed']
    ledger=[]
    def add(name,seed_value,mode='A',**extra):
        ledger.append({'id':name,'seed':seed_value,'mode':mode,'status':'NOT RUN','RUN_ID':None,**extra})
    add('E0',seed)
    add('E1-A-1',seed)
    add('E1-A-2',seed)
    for i in range(5):
        add(f'E2-{i+1}',seed+i)
    add('E3-A-continuous',seed)
    add('E3-A-resume',seed,restart_at_epoch=cfg['train']['epochs']//2,requires_new_process=True)
    add('CF02-workers-1',seed,mode='C',num_workers=2)
    add('CF02-workers-2',seed,mode='C',num_workers=2)
    add('CF03-weights-only',seed,ablation='model state only; explicit controlled diagnostic, not full resume')
    for mode in ['B','C']:
        for name in ['E1-1','E1-2','E3-continuous','E3-resume']:
            add(f'{name}-{mode}',seed,mode=mode,device='cuda')
    write_json(ROOT/'docs/environment/lab0_run_plan.json',{
        'budget_reference':budget,'cpu_and_worker_intervention_training_runs':13,'additional_cuda_runs':8,
        'counting_ambiguity':'CF02 workers=2 is mode C by PDF 9.2 although included in mode A total in 12.1',
        'ledger':ledger,'E4':{'status':'NOT RUN','batch_sizes':cfg['profile_batch_sizes'],'warmup':10,'repeats':30},
        'automatic_execution':False,
        'additional_measurements':'CF data-byte mutation, cold/warm profiling, first-batch checksums and CF4 near-equal final accuracy search still required'})
    print('Run plan saved; no formal training started.')


if __name__=='__main__':
    main()
