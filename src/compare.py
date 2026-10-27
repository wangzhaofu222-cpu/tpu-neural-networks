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
            'parameters':manifest['parameters'], **summary.get('trainability', {})})
    return {'groups':groups,'config_differences_from_first':[diff_configs(configs[0],c) for c in configs]}


def markdown_table(result):
    lines=['| RUN_ID | EXP_ID | Seed | ValAcc | ValLoss | Time s | Params | Status |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    def value(x):
        return '—' if x is None else f'{x:.6g}' if isinstance(x,float) else str(x)
    for exp_id,rows in result['groups'].items():
        for row in rows:
            fields=[row['RUN_ID'],exp_id,row['seed'],row['validation_accuracy'],row['validation_loss'],
                    row['seconds'],row['parameters'],row['status']]
            lines.append('| '+' | '.join(value(x) for x in fields)+' |')
    diagnostic_rows = [r for rows in result['groups'].values() for r in rows if 'rho_T' in r]
    if diagnostic_rows:
        lines.extend(['', '| RUN_ID | rho_T | GI | AVD | Numerical failure |', '|---|---:|---:|---:|---|'])
        for r in diagnostic_rows:
            lines.append('| '+' | '.join(value(r.get(k)) for k in ['RUN_ID', 'rho_T', 'GI', 'AVD', 'numerical_failure'])+' |')
    lines.extend(['','Config differences from the first run:',
                  json.dumps(result['config_differences_from_first'],ensure_ascii=False,indent=2)])
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--run-root',default='runs')
    p.add_argument('--format',choices=['json','markdown'],default='json')
    a = p.parse_args()
    result=compare_runs(a.runs,a.run_root)
    print(markdown_table(result) if a.format=='markdown' else json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
