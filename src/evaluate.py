import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.config import load_config
from src.data.vision import datasets
from src.engine.checkpoint import load_checkpoint
from src.models.small_cnn import build_model
from src.metrics.classification import evaluate_loader
from src.utils.reproducibility import seed_all
from src.utils.io import write_json


def evaluate_run(run,role='val',checkpoint='last.pt',run_root='runs'):
    path = Path(run) if Path(run).is_dir() else Path(run_root)/run
    cfg = load_config(path/'config.resolved.yaml')
    seed_all(cfg['experiment']['seed'],cfg['experiment']['deterministic'],cfg['runtime']['threads'])
    device = torch.device(cfg['runtime']['device'])
    ds,info = datasets(cfg)
    model = build_model(cfg['model']).to(device)
    state = load_checkpoint(path/'checkpoints'/checkpoint,model)
    if state['data_fingerprint'] != info['data_fingerprint'] or state['split_fingerprint'] != info['split_fingerprint']:
        raise ValueError('Evaluation data changed')
    if role not in ['val','test']:
        raise PermissionError('Only val/test evaluation is permitted here')
    result = evaluate_loader(model,DataLoader(ds[role],batch_size=cfg['train']['batch_size'],shuffle=False,num_workers=0),device)
    result.update(role=role,checkpoint=checkpoint,RUN_ID=path.name)
    # Evaluations are append-only artifacts and never replace training summaries.
    import uuid
    write_json(path/f'evaluation-{uuid.uuid4().hex}.json',result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--run-root',default='runs')
    p.add_argument('--role',choices=['val','test'],default='val')
    p.add_argument('--checkpoint',choices=['last.pt','best.pt'],default='last.pt')
    p.add_argument('--lab2-corruption',choices=['clean','gaussian_noise','blur','contrast','occlusion','translation','patch_shuffle'])
    p.add_argument('--level',type=int,choices=[0,1,2,3],default=0)
    a = p.parse_args()
    if a.lab2_corruption is not None:
        from src.engine.runner import ExperimentRunner
        print(ExperimentRunner.evaluate_checkpoint(a.run,a.lab2_corruption,a.level))
        return
    print(json.dumps(evaluate_run(a.run,a.role,a.checkpoint,a.run_root)))


if __name__ == '__main__':
    main()
