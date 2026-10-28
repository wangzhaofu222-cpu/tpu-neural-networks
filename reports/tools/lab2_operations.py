"""Administrative Lab 2 operations outside the frozen training-source scope."""
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime,timezone
from src.utils.io import write_json,file_hash

ROOT=Path('experiments/lab2')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def git(*args):return subprocess.run(['git','-c','safe.directory=D:/神经网络',*args],capture_output=True,text=True,encoding='utf-8',check=True).stdout.strip()
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['finalize-hypothesis','seal','all-corruptions','status']);p.add_argument('--source');a=p.parse_args()
    if a.stage=='finalize-hypothesis':
        selected=read(ROOT/'selection.json');lr=read(ROOT/'selected_lr.json');d=selected['validation_delta'];r=selected['fraction']
        original=git('show','a49fccf3907e64b00e1cf83b05ce755fcf126e80:experiments/lab2/hypothesis.md')
        (ROOT/'hypothesis.initial.md').write_text(original+'\n',encoding='utf-8')
        text=f'''# Lab 2 E3 finalized hypothesis

The conditional selection rule was committed in
a49fccf3907e64b00e1cf83b05ce755fcf126e80 before formal E1/LR search.
The original text is preserved as hypothesis.initial.md and in Git history.
E1 seed1 is explicitly exploratory and reused, as required by PDF9.3;
only seeds2,3 are new confirmatory training. No test measurement chooses
this regime: selection.json is a timestamped validation-only lock.

Comparison: canonical CIFAR ResNet18 versus ViT P4/L6/D384/H6/MLP1536,
parameter-matched mode, fixed fraction r={r:g}. Fixed balanced subset
indices from base_seed, common training seeds1,2,3, S10000 actual updates,
B128, A1, AdamW wd.05, warmup500 then cosine, float32, no pretraining.
CNN LR={lr['cnn']['lr']}, ViT LR={lr['vit']['lr']}, independently selected
once on r1 final validation from the preregistered three-point grid.

Expected mean test difference ViT minus CNN: approximately {d:.6f}
({100*d:.3f} percentage points), with {'positive' if d>0 else 'negative'} sign,
based on the selected seed1 final-validation gap, equation2.8. This is
a prediction, not an observed test mean. Locality and shared spatial
weights in equations2.1–2.2 versus learned global attention and positional
embeddings in2.3–2.5 motivate possible differences at fixed finite data/S.
No causal attribution to a single mechanism is claimed.

Refutation: an opposite/nonmatching mean sign, mixed or zero paired signs,
or abs(mean_delta)<2*sample_std_delta (ddof1) rejects the stated stable
architecture difference. All three paired differences must be shown;
uncertainty is reported even if it contradicts this expectation.

Seed1 source runs:
CNN {selected['runs']['cnn']['RUN_ID']}
ViT {selected['runs']['vit']['RUN_ID']}

The subset is held fixed across seeds. Three seeds describe initialization
and batch ordering variability; they do not cover new subset draws,
alternative recipes, or natural distribution shifts. E2 test observations
cannot alter the validation-locked fraction, LR, seed list or stopping.
'''
        (ROOT/'hypothesis.md').write_text(text,encoding='utf-8');return
    if a.stage=='seal':
        hyp=ROOT/'hypothesis.md';commit=git('rev-parse','HEAD')
        assert git('show',commit+':'+hyp.as_posix())==hyp.read_text(encoding='utf-8').strip()
        write_json(ROOT/'preregistration_evidence.json',{'git_commit':commit,'hypothesis_sha256':file_hash(hyp),
            'initial_conditional_commit':'a49fccf3907e64b00e1cf83b05ce755fcf126e80',
            'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'selected_fraction':read(ROOT/'selection.json')['fraction'],
            'new_seeds_not_yet_launched':[2,3],'seed1_reused_exploratory':True});return
    if a.stage=='all-corruptions':
        if not a.source:raise ValueError('--source training run directory required')
        from scripts.lab2_evaluate import evaluate
        from src.diagnostics.corruptions import LEVELS
        source=Path(a.source)
        labels=[v['label'] for v in read(ROOT/'ledger.json') if v['RUN_ID']==source.name and v['label'].startswith('E1_')]
        prefix=labels[0] if labels else f'all_{source.name}'
        evaluate(f'{prefix}_clean',source)
        for c in LEVELS:
            for s in range(4):evaluate(f'{prefix}_{c}_{s}',source,c,s)
        import csv
        rows=read(ROOT/'evaluations.json');wanted={f'{prefix}_{c}_{s}' for c in LEVELS for s in range(4)}|{f'{prefix}_clean'}
        with (ROOT/f'all_corruptions_{source.name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['label','corruption','level','accuracy','loss','count','parent_RUN_ID','RUN_ID','input_sha256'])
            w.writeheader();w.writerows({k:v[k] for k in w.fieldnames} for v in rows if v['label'] in wanted)
        return
    rows=read(ROOT/'ledger.json');training=[r for r in rows if r['kind']=='training' and not r['label'].startswith(('smoke','budget'))]
    active=[]
    for path in Path('runs/lab2').glob('*/manifest.json'):
        m=read(path)
        if m['status']=='RUNNING':
            stepfile=path.parent/'steps.jsonl';last=json.loads(stepfile.read_text().splitlines()[-1]) if stepfile.exists() else {}
            active.append({'name':yaml_config(path.parent)['experiment']['name'],'RUN_ID':m['RUN_ID'],'step':last.get('global_step',0),'target':10000})
    print(json.dumps({'formal_training_completed':len(training),'active':active},ensure_ascii=False))

def yaml_config(path):
    import yaml
    return yaml.safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8'))

if __name__=='__main__':main()
