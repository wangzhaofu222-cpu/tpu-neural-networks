import argparse
import json
from pathlib import Path
from src.config import load_config


def flatten(obj,prefix=''):
    result = {}
    for key,value in obj.items():
        key = prefix+key
        if isinstance(value,dict):
            result.update(flatten(value,key+'.'))
        else:
            result[key] = value
    return result


def diff_configs(a,b):
    aa,bb = flatten(a),flatten(b)
    return {k:[aa.get(k),bb.get(k)] for k in sorted(aa.keys()|bb.keys()) if aa.get(k)!=bb.get(k)}


def compare_runs(runs,root='runs'):
    groups = {}
    configs = []
    for run in runs:
        path = Path(run) if Path(run).is_dir() else Path(root)/run
        manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
        summary = json.loads((path/'summary.json').read_text(encoding='utf-8')) if (path/'summary.json').exists() else {}
        configs.append(load_config(path/'config.resolved.yaml'))
        groups.setdefault(manifest['EXP_ID'],[]).append({'RUN_ID':manifest['RUN_ID'],'seed':manifest['seed'],
            'status':manifest['status'],'validation_accuracy':summary.get('validation_metric'),
            'validation_loss':summary.get('validation_loss'),'seconds':manifest.get('wall_seconds'),
            'parameters':manifest['parameters']})
    return {'groups':groups,'config_differences_from_first':[diff_configs(configs[0],c) for c in configs]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--run-root',default='runs')
    a = p.parse_args()
    print(json.dumps(compare_runs(a.runs,a.run_root),indent=2))


if __name__ == '__main__':
    main()
