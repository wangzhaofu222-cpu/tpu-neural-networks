"""Serial orchestration only. Every forward/training run uses ExperimentRunner."""
import argparse
import copy
import json
import time
from pathlib import Path
from src.config import load_config
from src.engine.runner import ExperimentRunner
from src.utils.io import write_json

ROOT = Path('experiments/lab1')


def execute(label, cfg, probe=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    ledger = ROOT/'ledger.json'
    records = json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else []
    if any(r['label'] == label for r in records):
        return next(Path(r['path']) for r in records if r['label'] == label)
    cfg = copy.deepcopy(cfg)
    cfg['experiment']['name'] = label
    start = time.perf_counter()
    runner = ExperimentRunner(cfg)
    path = runner.run_probe() if probe else runner.run()
    record = {'label': label, 'path': path.as_posix(), 'RUN_ID': path.name, 'probe_only': probe,
              'elapsed_seconds_including_setup': time.perf_counter()-start}
    records.append(record)
    write_json(ledger, records)
    print(json.dumps({'finished': label, **record}), flush=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['smoke','budget','e1','lr','e2','boundary','e3','cf2'])
    args = parser.parse_args()
    base = load_config('configs/lab1.yaml')
    base['experiment']['purpose'] = 'formal'
    depths = [8,24,48,96]
    if args.stage in ['smoke','budget']:
        base['experiment']['purpose'] = args.stage
        base['train']['epochs'] = 1
        base['data']['train_fraction'] = .01 if args.stage == 'smoke' else 1.
        base['runtime']['max_seconds'] = 600
        if args.stage == 'smoke':
            for name in ['deep_mlp','residual_mlp']:
                base['model'].update(name=name,depth=8)
                execute('smoke_'+name, base)
        else:
            for name,depth in [('deep_mlp',8),('residual_mlp',96)]:
                base['model'].update(name=name,depth=depth)
                execute('budget_'+name+str(depth),base)
        return
    if args.stage == 'e1':
        for init in ['he','xavier','gaussian']:
            for depth in depths:
                base['model'].update(initialization=init,depth=depth)
                execute(f'E1_{init}_{depth}',base,probe=True)
    elif args.stage == 'lr':
        for lr in [.001,.003,.01,.03,.1]:
            base['optimizer']['lr'] = lr
            execute(f'LR_{lr}',base)
        records = json.loads((ROOT/'ledger.json').read_text())
        candidates = []
        for record in records:
            if record['label'].startswith('LR_'):
                summary = json.loads((Path(record['path'])/'summary.json').read_text())
                if summary['status'] == 'COMPLETED':
                    candidates.append((summary['validation_metric'], -summary['validation_loss'],
                                      -float(record['label'][3:]),record['RUN_ID']))
        best = max(candidates)
        write_json(ROOT/'selected_lr.json', {'lr': -best[2], 'RUN_ID': best[3],
                   'rule': 'maximum final validation accuracy; ties minimum validation loss then smaller LR; 3 full epochs'})
    else:
        base['optimizer']['lr'] = json.loads((ROOT/'selected_lr.json').read_text())['lr']
        if args.stage == 'e2':
            for architecture in ['plain','residual_unit','residual_scaled']:
                for depth in depths:
                    base['model'].update(name='deep_mlp' if architecture=='plain' else 'residual_mlp',
                              alpha=1. if architecture=='residual_unit' else 'inverse_sqrt_depth',depth=depth)
                    execute(f'E2_{architecture}_{depth}',base)
        elif args.stage == 'boundary':
            base['model']['initialization'] = 'gaussian'
            base['train']['epochs'] = 1
            for depth in depths:
                base['model']['depth'] = depth
                execute(f'E1_after_epoch_gaussian_{depth}',base)
        elif args.stage == 'cf2':
            base['model'].update(initialization='gaussian',width=1024,depth=96)
            execute('CF2_gaussian_width1024_depth96',base,probe=True)
        elif args.stage == 'e3':
            prereg = json.loads((ROOT/'e3_preregistration.json').read_text())
            from src.utils.provenance import git_state
            import subprocess
            code = git_state()
            if code['dirty']:
                raise RuntimeError('Uncommitted E3 source')
            committed = subprocess.run(['git','-c','safe.directory=D:/神经网络','show','HEAD:hypothesis.md'],
                                       capture_output=True,encoding='utf-8',check=True).stdout
            if committed != Path('hypothesis.md').read_text(encoding='utf-8'):
                raise RuntimeError('hypothesis.md is not committed exactly')
            for arm in ['A','B']:
                base['model'].update(prereg[arm])
                for seed in [1,2,3]:
                    base['experiment']['seed'] = seed
                    execute(f'E3_{arm}_seed{seed}',base)


if __name__ == '__main__':
    main()
