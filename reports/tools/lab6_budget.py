import copy,json,time
from pathlib import Path
import yaml,torch
from src.engine.runner import ExperimentRunner
from src.utils.io import write_json

def main():
    base=yaml.safe_load(Path('configs/lab6.yaml').read_text(encoding='utf-8'));rows=[]
    for name,depth,threads in [('energy_model',3,1),('energy_model',3,2),('normalizing_flow',2,2),('normalizing_flow',16,2)]:
        c=copy.deepcopy(base);c['experiment'].update(purpose='budget',role='budget')
        c['model'].update(name=name,depth=depth,width=128 if name=='energy_model' else 64)
        c['runtime']['threads']=threads;c['train']['max_steps']=30;c['logging']['keep_epochs']=[]
        start=time.perf_counter();p=ExperimentRunner(c).run();s=json.loads((p/'summary.json').read_text())
        seconds=s['training_seconds'];rows.append({'model':name,'depth':depth,'threads':threads,'device':'cpu','RUN_ID':p.name,
            'updates':30,'K':50,'training_seconds':seconds,'wall_seconds':time.perf_counter()-start,'seconds_per_update':seconds/30,
            'projected_5000_update_seconds':seconds/30*5000,'status':s['status']})
        write_json('experiments/lab6/budget.json',{'measurements':rows,'formal_device':'cpu','formal_threads':2,
            'note':'30-update budget runs are not formal experiment results; all formal models must execute5000updates'})
        print(rows[-1],flush=True)
if __name__=='__main__':main()
