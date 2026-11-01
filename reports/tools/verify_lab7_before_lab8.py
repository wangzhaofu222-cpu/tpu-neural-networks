"""Read-only historical replay into a separate verification directory."""
import json, shutil, subprocess
from pathlib import Path
from src.utils.io import file_hash, write_json

def main():
    original=Path('experiments/lab7'); target=Path('experiments/lab7_verification/replayed')
    target.parent.mkdir(parents=True,exist_ok=True)
    protected={str(p):file_hash(p) for root in [Path('experiments'),Path('runs'),Path('configs'),Path('data/splits')] if root.exists() for p in root.rglob('*') if p.is_file() and 'lab7_verification' not in p.parts}
    write_json(target.parent/'historical_files.json',protected)
    shutil.copytree(original,target,dirs_exist_ok=True)
    import reports.tools.lab7_artifacts as plots
    plots.R=target; plots.F=target/'figures'; plots.main()
    regenerated=[{'file':str(p.relative_to(target)),'identical':file_hash(p)==file_hash(original/p.relative_to(target))} for p in list(target.glob('*.csv'))+list((target/'figures').glob('*'))]
    import reports.tools.lab7_audit as audit
    audit.R=target; audit.main()
    result=json.loads((target/'final_audit.json').read_text())
    changed=[p for p,h in protected.items() if not Path(p).exists() or file_hash(p)!=h]
    report={'status':'PASS' if result['status']=='PASS' and not changed and all(x['identical'] for x in regenerated) else 'FAIL','audit_checks':result['check_count'],'failed':result['failed'],'checkpoints':result['statistics']['all_checkpoints'],'regeneration':regenerated,'changed_historical_files':changed,'HEAD':subprocess.check_output(['git','-c','safe.directory=D:/神经网络','rev-parse','HEAD']).decode().strip(),'optimizer_updates':0}
    write_json(target.parent/'audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='regeneration'},ensure_ascii=False))

if __name__=='__main__': main()
