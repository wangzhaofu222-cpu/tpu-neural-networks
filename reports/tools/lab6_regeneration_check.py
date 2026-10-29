"""Regenerate tables/plots in a fresh verification folder, without model training."""
import shutil,json
from pathlib import Path
from src.utils.io import file_hash,write_json

def main():
    root=Path('experiments/lab6');target=root/'verification/regenerated'
    if target.exists():raise RuntimeError('Refusing to overwrite previous verification')
    target.mkdir(parents=True)
    for name in ['E1.json','E2.json','E3.json','run_index.json']:shutil.copy2(root/name,target/name)
    shutil.copytree(root/'noise',target/'noise')
    import reports.tools.lab6_artifacts as artifacts
    artifacts.R=target;artifacts.F=target/'figures';artifacts.main()
    names=['E1_eta.csv','E1_quadratic.csv','E2_depths.csv','E3_models.csv','E3_pairs.csv','counterfactual_answers.json','counterfactual_answers.md']
    records=[{'file':name,'byte_identical':file_hash(root/name)==file_hash(target/name)} for name in names]
    images=[{'file':p.name,'byte_identical':file_hash(p)==file_hash(target/'figures'/p.name)} for p in sorted((root/'figures').glob('*.png'))]
    result={'status':'PASS' if all(r['byte_identical'] for r in records+images) else 'FAIL','tables_and_answers':records,'PNG_previews':images,
        'note':'SVG timestamps are not compared; scientific tables and PNG previews are compared byte-for-byte; no optimizer updates.'}
    write_json(root/'regeneration_audit.json',result);print(json.dumps(result))
if __name__=='__main__':main()
