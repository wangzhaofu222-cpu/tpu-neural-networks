"""Independent Lab4 recomputation in a fresh destination; preserve historical evidence."""
import json
import shutil
from pathlib import Path
from src.utils.io import file_hash,write_json

OUT=Path('experiments/lab4_verification')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    OUT.mkdir(exist_ok=True);target=OUT/'recomputed'
    if target.exists():raise RuntimeError('Refusing overwrite of a previous verification')
    roots=[f'experiments/lab{i}' for i in range(5)]+[f'runs/lab{i}' for i in range(5)]+['reports']
    hashes={p.as_posix():file_hash(p) for root in roots for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write_json(OUT/'historical_files.json',hashes);print('Historical files protected:',len(hashes),flush=True)
    shutil.copytree('experiments/lab4',target)
    import reports.tools.lab4_results as results
    results.ROOT=target;results.FIG=target/'figures';results.main()
    import reports.tools.lab4_audit as audit
    audit.ROOT=target;audit.main()
    old,new=read('experiments/lab4/results.json'),read(target/'results.json')
    assert old==new,'Recomputed scientific results differ'
    for name in ['E1.csv','E1_probe_checkpoints.csv','E2.csv','E3.csv','E3_gain.csv','counterfactuals.json']:
        assert file_hash(target/name)==file_hash(Path('experiments/lab4')/name),name
    assert len(read(target/'checkpoint_index.json'))==334
    assert new['counterfactual_status']['Q2']=='BLOCKED'
    assert all(Path(p).exists() and file_hash(p)==h for p,h in hashes.items())
    write_json(OUT/'audit.json',{'status':'PASS','original_records_unchanged':True,'protected_files':len(hashes),
        'checkpoint_count':334,'formal_checkpoint_count':325,'statistics_recomputed':True,'Q2':'BLOCKED',
        'preregistration_commit':read(target/'preregistration_evidence.json')['git_commit'],
        'final_Lab4_commit':'709e39cf7d7bc8ab4759c043c1795e063356d292'})
    print('Lab4 independent verification PASS',flush=True)
if __name__=='__main__':main()
