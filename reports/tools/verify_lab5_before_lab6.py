"""Regenerate Lab5 evidence separately, without retraining or touching old RUNs."""
import json,shutil
from pathlib import Path
from src.utils.io import write_json,file_hash

OUT=Path('experiments/lab5_verification')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    OUT.mkdir(exist_ok=True);target=OUT/'recomputed'
    if target.exists():raise RuntimeError('Refusing to overwrite prior independent verification')
    roots=[f'experiments/lab{i}' for i in range(6)]+[f'runs/lab{i}' for i in range(6)]+['reports']
    hashes={p.as_posix():file_hash(p) for root in roots for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write_json(OUT/'historical_files.json',hashes);print('Historical files protected:',len(hashes),flush=True)
    shutil.copytree('experiments/lab5',target)
    import reports.tools.lab5_artifacts as artifacts
    artifacts.R=target;artifacts.F=target/'figures';artifacts.main()
    import reports.tools.lab5_audit as audit
    audit.R=target;audit.main()
    new=read(target/'final_audit.json');assert new['status']=='PASS' and new['check_count']==625
    for name in ['E1_models.csv','E1_deletion.csv','E2_pairs.csv','E3_depths.csv','E3_all_layers.csv','structure_map.csv','counterfactual_answers.md']:
        assert file_hash(target/name)==file_hash(Path('experiments/lab5')/name),name
    stats=new['statistics'];assert (stats['formal_checkpoints'],stats['all_lab5_checkpoints'])==(274,286)
    assert new['scientific_predictions']['E2_preregistered_magnitude_band'].startswith('FAIL')
    assert not new['teacher_split_independently_verified'] and new['Lab4_Q2'].startswith('BLOCKED')
    assert all(Path(p).exists() and file_hash(p)==h for p,h in hashes.items())
    write_json(OUT/'audit.json',{'status':'PASS','Lab5_final_commit':'bc758319fe70b66bc066667924c8f5554629cac4',
        'protected_files':len(hashes),'original_records_unchanged':True,'audit_checks':625,'formal_checkpoints':274,'all_checkpoints':286,
        'E2_magnitude_prediction':'FAIL (retained)','Lab4_Q2':'BLOCKED','Lab5_course_split':'UNVERIFIED',
        'regenerated_tables_byte_identical':True,'no_formal_retraining':True})
    print('Lab5 independent verification PASS',flush=True)
if __name__=='__main__':main()
