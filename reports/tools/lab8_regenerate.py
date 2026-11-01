"""Rebuild only derived Lab8 artifacts in a separate namespace."""
import shutil
from pathlib import Path
from src.utils.io import file_hash,write_json

def main():
    source=Path('experiments/lab8');out=Path('tmp/lab8_regenerated');out.mkdir(exist_ok=True,parents=True)
    for p in source.glob('*.json'):shutil.copy2(p,out/p.name)
    import reports.tools.lab8_artifacts as artifacts
    artifacts.R=out;artifacts.F=out/'figures';artifacts.main()
    import reports.tools.lab8_finalize as answers
    answers.R=out;answers.main()
    names=[str(p.relative_to(out)) for p in out.glob('*.csv')]+[str(p.relative_to(out)) for p in (out/'figures').glob('*')]+['counterfactual_answers.json','counterfactual_answers.md']+[str(p.relative_to(out)) for p in (out/'profiles').rglob('reliability.json')]
    rows=[{'file':n,'byte_identical':file_hash(source/n)==file_hash(out/n)} for n in names]
    write_json(source/'regeneration_audit.json',{'status':'PASS' if all(r['byte_identical'] for r in rows) else 'FAIL','checks':rows,'count':len(rows),'classifier_inference_calls':0,'training_updates':0})
    print('Regenerated artifacts',len(rows),'mismatches',[r for r in rows if not r['byte_identical']],flush=True)

if __name__=='__main__':main()
