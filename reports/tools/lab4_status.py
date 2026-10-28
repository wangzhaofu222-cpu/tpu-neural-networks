"""Compact read-only progress query for the serial Lab4 protocol."""
import json
from pathlib import Path
import yaml

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    root=Path('experiments/lab4')
    active=[]
    for path in Path('runs/lab4').iterdir():
        if not (path/'manifest.json').exists():continue
        m=read(path/'manifest.json')
        if m['status']!='RUNNING':continue
        summary=read(path/'summary.json') if (path/'summary.json').exists() else {}
        cfg=yaml.safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8'))
        active.append({'RUN_ID':path.name,'role':cfg['experiment']['role'],'epochs':cfg['train']['epochs'],
            'epoch':summary.get('epoch'),'loss':summary.get('train_loss')})
    completed=[]
    if (root/'ledger.json').exists():
        completed=[{'label':r['label'],'status':r['status']} for r in read(root/'ledger.json')]
    probes=[]
    if (root/'evaluations.json').exists():
        probes=[{'label':r['label'],'probe':r['test_accuracy'],'knn':r['knn_test_accuracy']}
            for r in read(root/'evaluations.json') if r['label']!='budget_probe']
    print(json.dumps({'active':active,'completed_training':completed,'probe_count':len(probes),
        'recent_probes':probes[-3:]},ensure_ascii=False))

if __name__=='__main__':main()
