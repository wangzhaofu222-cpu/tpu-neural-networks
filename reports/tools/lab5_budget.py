import json,time
from pathlib import Path
from src.graph.experiments import config,run_train
from src.utils.io import write_json

def main():
    rows=[]
    for device,kind,depth in [('cpu','mlp',2),('cpu','gcn',2),('cpu','gat',2),('cpu','gcn',24),('cuda','gcn',2),('cuda','gat',2)]:
        c=config(kind=kind,depth=depth,purpose='budget',role='budget')
        c['runtime']['device']=device;c['train']['epochs']=5
        start=time.perf_counter()
        try:
            p=Path(run_train(c));s=json.loads((p/'summary.json').read_text());metrics=[json.loads(v) for v in (p/'metrics.jsonl').read_text().splitlines()]
            rows.append({'device':device,'kind':kind,'depth':depth,'RUN_ID':p.name,'status':s['status'],
               'training_seconds':s['training_seconds'],'wall_seconds':time.perf_counter()-start,
               'steady_epoch_seconds':sum(r['epoch_time'] for r in metrics[1:])/4,
               'projected_200_epoch_seconds':sum(r['epoch_time'] for r in metrics[1:])*50,
               'peak_gpu_bytes':metrics[-1]['peak_gpu_memory_bytes']})
        except Exception as e:rows.append({'device':device,'kind':kind,'depth':depth,'status':'FAILED','error':str(e)})
        write_json('experiments/lab5/budget.json',{'measurements':rows,'formal_device':'cpu','decision':'CPU full-batch deterministic; no GPU needed by teacher. Budget runs are not formal evidence.'})
        print(rows[-1],flush=True)
if __name__=='__main__':main()
