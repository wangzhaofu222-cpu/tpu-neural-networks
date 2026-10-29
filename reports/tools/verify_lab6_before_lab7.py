"""Recheck frozen Lab6 in a separate output namespace, without optimization."""
import json
import shutil
from pathlib import Path
import torch
import yaml
from src.utils.io import write_json, file_hash, digest
from src.models.small_cnn import build_model

OUT = Path('experiments/lab6_verification')


def main():
    OUT.mkdir(exist_ok=True)
    target = OUT / 'recomputed'
    if target.exists():
        raise RuntimeError('Refusing to overwrite existing verification')
    roots = [f'experiments/lab{i}' for i in range(7)] + [f'runs/lab{i}' for i in range(7)]
    protected = {p.as_posix(): file_hash(p) for root in roots for p in Path(root).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts}
    write_json(OUT / 'historical_files.json', protected)
    print('Historical files protected:', len(protected), flush=True)
    shutil.copytree('experiments/lab6', target)
    import reports.tools.lab6_audit as audit
    audit.R = target
    audit.main()
    result = json.loads((target / 'final_audit.json').read_text(encoding='utf-8'))
    assert result['status'] == 'PASS' and result['check_count'] == 234
    import reports.tools.lab6_artifacts as artifacts
    artifacts.R = target
    artifacts.F = target / 'figures'
    artifacts.main()
    names = ['E1_eta.csv', 'E1_quadratic.csv', 'E2_depths.csv', 'E3_models.csv',
             'E3_pairs.csv', 'counterfactual_answers.json', 'counterfactual_answers.md']
    names += ['figures/' + p.name for p in Path('experiments/lab6/figures').glob('*.png')]
    assert all(file_hash(target / n) == file_hash(Path('experiments/lab6') / n) for n in names)
    rows = []
    for f in sorted(Path('runs/lab6').glob('*/checkpoints/*.pt')):
        cfg = yaml.safe_load((f.parent.parent / 'config.resolved.yaml').read_text(encoding='utf-8'))
        state = torch.load(f, map_location='cpu', weights_only=False)
        model = build_model(cfg['model'])
        model.load_state_dict(state['model'], strict=True)
        assert state['config_hash'] == digest(cfg)
        rows.append({'path': str(f), 'sha256': file_hash(f), 'RUN_ID': f.parent.parent.name,
                     'seed': cfg['experiment']['seed'], 'status': 'PASS'})
    assert len(rows) == 63
    assert all(Path(p).exists() and file_hash(p) == h for p, h in protected.items())
    write_json(OUT / 'checkpoint_loadability.json', rows)
    write_json(OUT / 'audit.json', {'status': 'PASS', 'audit_checks': 234,
        'protected_files': len(protected), 'all_checkpoints_loaded': len(rows),
        'formal_checkpoints': 55, 'tables_answers_PNG_byte_identical': True,
        'original_records_unchanged': True, 'optimizer_updates': 0,
        'Lab6_E1_mixing': 'FAIL retained', 'Lab4Q2': 'BLOCKED',
        'Lab5_course_split': 'UNVERIFIED', 'Lab5_E2_magnitude': 'FAIL retained',
        'Lab6_final_commit': '23f049d657db2d0dc1b33e68a384299fe961d886'})
    print('Lab6 independent verification PASS', flush=True)


if __name__ == '__main__':
    main()
