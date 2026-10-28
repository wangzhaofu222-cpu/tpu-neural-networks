"""Read-only independent audit; recomputation writes only to a new audit directory."""
import json
import shutil
from pathlib import Path
import torch
import yaml
from src.utils.io import digest, file_hash, write_json
from src.data.splits import train_subset, validate_split

OUT = Path('experiments/lab2_verification')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    OUT.mkdir(exist_ok=True)
    protected = []
    for root in ['experiments/lab0','experiments/lab1','experiments/lab1_verification','experiments/lab2','runs/lab0','runs/lab1','runs/lab2','reports']:
        protected.extend(p for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in str(p))
    hashes = {p.as_posix(): file_hash(p) for p in protected}
    write_json(OUT/'historical_files.json', hashes)
    print(f'Snapshotted {len(hashes)} historical files', flush=True)
    checks=[]
    torch.set_num_threads(2)
    for row in read('experiments/lab2/checkpoint_index.json'):
        p=Path(row['path']); parent=p.parent.parent
        cfg=yaml.safe_load((parent/'config.resolved.yaml').read_text(encoding='utf-8'))
        m=read(parent/'manifest.json'); st=torch.load(p,map_location='cpu',weights_only=False)
        assert file_hash(p)==row['sha256']
        assert digest(cfg)==m['config_hash']==st['config_hash']
        assert st['source_hash']==m['source_hash'] and st['split_fingerprint']==m['split_fingerprint']
        assert cfg['experiment']['seed']==m['seed']==row['seed']
        assert 0<st['global_step']<=row['steps']
        assert all(torch.isfinite(v).all() for v in st['model'].values() if v.is_floating_point())
        from src.models.small_cnn import build_model
        model=build_model(cfg['model']); model.load_state_dict(st['model'],strict=True)
        checks.append(dict(path=p.as_posix(),global_step=st['global_step'],status='PASS'))
        del model,st
    print(f'Loaded and verified {len(checks)} checkpoints',flush=True)
    ledger=read('experiments/lab2/ledger.json')
    from torchvision.datasets import CIFAR10
    example=next(r for r in ledger if r['label']=='E1_cnn_r1_j1')
    cfg=yaml.safe_load((Path(example['path'])/'config.resolved.yaml').read_text(encoding='utf-8'))
    split=read(cfg['data']['split_path']);validate_split(split)
    labels=CIFAR10(cfg['data']['root'],train=True,download=False).targets
    subsets={}
    for r in ledger:
        if not r['label'].startswith('E1_'): continue
        m=read(Path(r['path'])/'manifest.json'); c=yaml.safe_load((Path(r['path'])/'config.resolved.yaml').read_text(encoding='utf-8'))
        expected=train_subset(labels,split['roles']['train'],c['data']['train_fraction'],c['profile']['base_seed'])
        assert expected==m['train_subset_indices'] and digest(expected)==m['subset_fingerprint']
        subsets[c['data']['train_fraction']]=set(expected)
    ordered=sorted(subsets)
    assert all(subsets[a]<=subsets[b] for a,b in zip(ordered,ordered[1:]))
    for arm in ['cnn','vit']:
        candidates=[r for r in ledger if r['label'].startswith('LR_'+arm+'_')]
        winner=max(candidates,key=lambda r:read(Path(r['path'])/'summary.json')['validation_metric'])
        assert winner['RUN_ID']==read('experiments/lab2/selected_lr.json')[arm]['RUN_ID']
    import reports.tools.lab2_results as recompute
    target=OUT/'recomputed'
    if target.exists(): raise RuntimeError('Audit output exists; refusing overwrite')
    shutil.copytree('experiments/lab2',target)
    recompute.ROOT=target;recompute.FIG=target/'figures';recompute.main()
    old=read('experiments/lab2/results.json');new=read(target/'results.json')
    for key in ['E1','E2','E3','deltas','E3_rows','formal_training_executions','formal_optimizer_updates','acceptance']:
        assert old[key]==new[key],key
    for name in ['E1.csv','E2.csv','E3.csv','trainability_map.csv','CF5_translations.csv','counterfactuals.json','interpretation.json']:
        assert file_hash(target/name)==file_hash(Path('experiments/lab2')/name),name
    assert all(Path(p).exists() and file_hash(p)==h for p,h in hashes.items())
    write_json(OUT/'audit.json',dict(status='PASS',protected_files=len(hashes),checkpoints=checks,
        nested_subsets='PASS',validation_only_lr_selection='PASS',raw_statistics_and_tables_reproduced='PASS',
        interruption_exact_overlap=read(target/'interruption.json')['overlapping_step_records_verified'],
        historical_files_unchanged=True,limitations=old['limitations'],E3=new['E3']))
    print('Lab2 independent audit PASS; original records unchanged',flush=True)

if __name__=='__main__': main()
