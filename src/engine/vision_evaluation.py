"""Append-only Lab 2 evaluations with checkpoint and identical-input provenance."""
import copy
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.config import load_config
from src.data.vision import datasets
from src.diagnostics.corruptions import CorruptedImages
from src.engine.checkpoint import load_checkpoint
from src.metrics.classification import evaluate_loader
from src.models.small_cnn import build_model
from src.utils.io import digest, file_hash, write_json
from src.utils.provenance import create_run, git_state
from src.utils.reproducibility import seed_all


def evaluate_checkpoint(source, corruption='clean', level=0):
    source=Path(source);cfg=load_config(source/'config.resolved.yaml');code=git_state()
    if cfg['experiment']['lab']!=2:raise ValueError('Lab 2 checkpoints only')
    if cfg['experiment']['purpose']=='formal' and code['dirty']:raise RuntimeError('Uncommitted source')
    if not Path('experiments/lab2/selection.json').exists():raise RuntimeError('Lock validation-only regime before test access')
    seed_all(cfg['experiment']['seed'],True,cfg['runtime']['threads'])
    ds,info=datasets(cfg);device=torch.device(cfg['runtime']['device'])
    checkpoint=source/'checkpoints/last.pt';model=build_model(cfg['model']).to(device)
    state=load_checkpoint(checkpoint,model)
    if state['data_fingerprint']!=info['data_fingerprint'] or state['split_fingerprint']!=info['split_fingerprint']:
        raise ValueError('Checkpoint data/split mismatch')
    key={'split':info['split_fingerprint'],'normalization':info['normalization'],
         'corruption':corruption,'level':level,'seed':cfg['profile']['base_seed'],'version':1}
    cache=Path('data/lab2-cache')/(digest(key)+'.pt');cache.parent.mkdir(exist_ok=True,parents=True)
    if not cache.exists():
        base=ds['test'] if corruption=='clean' else CorruptedImages(ds['test'],corruption,level,cfg['profile']['base_seed'])
        items=[base[i] for i in range(len(base))]
        x=torch.stack([v[0] for v in items]);y=torch.tensor([v[1] for v in items])
        torch.save({'x':x,'y':y,'key':key},cache)
    cached=torch.load(cache,map_location='cpu',weights_only=True)
    if cached['key']!=key:raise ValueError('Input cache metadata mismatch')
    # Hash actual normalized input bytes; s=0 and clean must hash identically.
    import hashlib
    input_hash=hashlib.sha256(cached['x'].numpy().tobytes()+cached['y'].numpy().tobytes()).hexdigest()
    cfg=copy.deepcopy(cfg);cfg['experiment'].update(name=f'eval_{source.name}_{corruption}_{level}',role='evaluation')
    cfg['evaluation']={'parent_RUN_ID':source.name,'checkpoint_sha256':file_hash(checkpoint),**key,'input_sha256':input_hash}
    info.update(evaluation=cfg['evaluation'])
    path,manifest=create_run(cfg,code,info,model,source.name);start=time.perf_counter()
    result=evaluate_loader(model,DataLoader(TensorDataset(cached['x'],cached['y']),batch_size=128,shuffle=False),device)
    result.update(RUN_ID=path.name,parent_RUN_ID=source.name,corruption=corruption,level=level,
                  input_sha256=input_hash,checkpoint_sha256=cfg['evaluation']['checkpoint_sha256'],
                  status='COMPLETED_EVALUATION',optimizer_steps=0,wall_seconds=time.perf_counter()-start)
    write_json(path/'summary.json',result)
    (path/'metrics.jsonl').write_text(json.dumps(result)+'\n',encoding='utf-8')
    manifest.update(status='COMPLETED_EVALUATION',global_step=0,completed_epoch=0,wall_seconds=result['wall_seconds'])
    write_json(path/'manifest.json',manifest)
    return path
