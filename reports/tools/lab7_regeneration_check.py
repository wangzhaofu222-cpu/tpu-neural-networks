"""Byte-check independent regeneration in a fresh namespace, without training."""
import shutil
from pathlib import Path
from src.utils.io import write_json,file_hash


def main():
    root=Path('experiments/lab7');out=root/'verification/regenerated'
    if out.exists():raise RuntimeError('Do not overwrite previous independent regeneration')
    out.mkdir(parents=True)
    for name in ['E1.json','E2.json','E3.json','diagnostics.json','supplements.json','counterfactual_answers.json']:
        shutil.copyfile(root/name,out/name)
    shutil.copytree(root/'metrics',out/'metrics')
    import reports.tools.lab7_artifacts as artifacts
    artifacts.R=out;artifacts.F=out/'figures';artifacts.main()
    from reports.tools.lab7_counterfactuals import render_markdown
    import json
    render_markdown(json.loads((out/'counterfactual_answers.json').read_text(encoding='utf-8')),out/'counterfactual_answers.md')
    paths=list(root.glob('*.csv'))+list((root/'figures').glob('*.png'))+list((root/'figures').glob('*.svg'))+[root/'counterfactual_answers.md']
    checks=[{'file':str(p.relative_to(root)),'byte_identical':file_hash(p)==file_hash(out/p.relative_to(root))} for p in paths]
    result={'status':'PASS' if all(r['byte_identical'] for r in checks) else 'FAIL','checks':checks,
            'optimizer_updates':0,'note':'SVGtimestamps suppressed andhashsalt fixed;tables/PNG/SVG compared byte-for-byte'}
    write_json(root/'regeneration_audit.json',result);print(result['status'],len(checks),'regenerated artifacts',flush=True)


if __name__=='__main__':main()
