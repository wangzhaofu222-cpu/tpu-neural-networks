"""Historical Lab8 verification in a separate namespace, no training."""
import shutil,json,subprocess
from pathlib import Path
from src.utils.io import file_hash,write_json

def main():
    out=Path('experiments/lab8_verification');target=out/'replayed';out.mkdir(exist_ok=True)
    roots=[Path(f'{kind}/lab{i}') for kind in ['experiments','runs'] for i in range(9)]
    protected={str(p):file_hash(p) for root in roots for p in root.rglob('*') if p.is_file()}
    write_json(out/'historical_files.json',protected)
    shutil.copytree('experiments/lab8',target,dirs_exist_ok=True)
    import reports.tools.lab8_audit as audit
    audit.R=target;audit.main()
    import reports.tools.lab8_artifacts as plots
    plots.R=target;plots.F=target/'figures';plots.main()
    import reports.tools.lab8_finalize as answers
    answers.R=target;answers.main()
    names=[str(p.relative_to(target)) for p in target.glob('*.csv')]+[str(p.relative_to(target)) for p in (target/'figures').glob('*')]+['counterfactual_answers.json','counterfactual_answers.md']+[str(p.relative_to(target)) for p in (target/'profiles').rglob('reliability.json')]
    regenerated=[{'file':n,'identical':file_hash(target/n)==file_hash(Path('experiments/lab8')/n)} for n in names]
    result=json.loads((target/'final_audit.json').read_text(encoding='utf-8'))
    changed=[p for p,h in protected.items() if not Path(p).exists() or file_hash(p)!=h]
    report={'status':'PASS' if result['status']=='PASS' and not changed and all(r['identical'] for r in regenerated) else 'FAIL','audit_checks':result['check_count'],'failed':result['failed'],'regenerated':regenerated,'changed_historical_files':changed,'checkpoint_count':len(json.loads((target/'checkpoint_inventory.json').read_text())),'optimizer_updates':0,'HEAD':subprocess.check_output(['git','-c','safe.directory=D:/神经网络','rev-parse','HEAD']).decode().strip()}
    write_json(out/'audit.json',report);print({k:v for k,v in report.items() if k!='regenerated'},flush=True)

if __name__=='__main__':main()
