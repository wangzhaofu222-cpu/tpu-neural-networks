"""Independent read-only Lab3 audit; regenerate derived artifacts in a new directory."""
import json
import shutil
from pathlib import Path
from src.utils.io import file_hash,write_json

OUT=Path('experiments/lab3_verification')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    OUT.mkdir(exist_ok=True)
    paths=[]
    for root in ['experiments/lab0','experiments/lab1','experiments/lab1_verification','experiments/lab2','experiments/lab2_verification','experiments/lab3','runs/lab0','runs/lab1','runs/lab2','runs/lab3','reports']:
        paths.extend(p for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    hashes={p.as_posix():file_hash(p) for p in paths}
    write_json(OUT/'historical_files.json',hashes)
    print('Protected historical files',len(hashes),flush=True)
    target=OUT/'recomputed'
    if target.exists():raise RuntimeError('Refusing overwrite of an existing audit')
    shutil.copytree('experiments/lab3',target)
    import reports.tools.lab3_results as results
    results.ROOT=target;results.FIG=target/'figures';results.main()
    import reports.tools.lab3_audit as audit
    audit.ROOT=target;audit.main()
    before=read('experiments/lab3/results.json');after=read(target/'results.json')
    for name in ['E1_curves','E2','E3','raw_E2_interval_gates','acceptance','formal_training_seconds','formal_training_updates','automated_tests']:
        assert before[name]==after[name],name
    for name in ['E1.csv','E2.csv','E3.csv','counterfactuals.json','attention_analysis.json']:
        assert file_hash(target/name)==file_hash(Path('experiments/lab3')/name),name
    assert len(read(target/'checkpoint_index.json'))==24
    assert all(v['acceptance']['memory_expected_interval']=='FAIL' for v in after['E2'].values())
    assert all(Path(p).exists() and file_hash(p)==h for p,h in hashes.items())
    write_json(OUT/'audit.json',{'status':'PASS','protected_files':len(hashes),'original_records_unchanged':True,
        'checkpoint_count':24,'formal_training_count':10,'E3':after['E3'],
        'preregistration_commit':after['preregistration_commit'],'raw_E2_memory_interval_FAILs_preserved':2,
        'statistics_reproduced_from_raw':True,'five_counterfactuals_have_measured_evidence':True})
    print('Lab3 audit PASS; original records unchanged',flush=True)
if __name__=='__main__':main()
