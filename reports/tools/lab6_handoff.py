"""Persist final supporting logs and append current scope without rewriting history."""
import json
import shutil
from pathlib import Path


def main():
    root = Path('experiments/lab6')
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    for name in ['budget', 'e1', 'e2', 'e3', 'final_tests', 'audit',
                 'regeneration', 'old_checkpoint_check', 'old_tests']:
        source = Path('tmp') / f'lab6_{name}.log'
        if source.exists():
            shutil.copyfile(source, logs / source.name)
    audit = json.loads((root / 'final_audit.json').read_text(encoding='utf-8'))
    block = f'''

<!-- LAB6_AUDITED_STATUS -->
## Lab6 当前完成状态

Lab6 已完成全部必做实验，复用既有 ExperimentRunner；11次正式模型训练各5000次更新。
153项自动测试通过，独立审计{audit['check_count']}项通过，5份CSV、两种反事实答案和8张PNG独立再生成逐字节一致。
当前需求状态为57 PASS、1 FAIL、0 BLOCKED、4 NOT_APPLICABLE。FAIL为E1实测中心间混合不足20%；已按教师后备规则选择eta=.02，未掩盖全部eta的Cov=0。
E2按val配对SEM选择Kf=8；E3三seed的MMD²差异满足教师统计判据。事前Git为214ea6627908fcbdf9d168b02b3bf59760375003。
完整中文结果、RUN_ID和表格见 [Lab6汇总](experiments/lab6/EXPERIMENT_SUMMARY.md)，
[逐项要求](docs/requirements/requirements_lab6.md)，[独立审计](experiments/lab6/final_audit.json)，
[复现说明](experiments/lab6/README.md)。正式训练累计1952.580秒；55个正式checkpoint，含预算共63个，本地保留且有哈希索引。
Lab5旧130测试、625审计及全部286个checkpoint加载复核通过，5450个历史文件未改；Lab5 E2量级预测FAIL、课程split UNVERIFIED、Lab4 Q2 BLOCKED均保留。
这些段落是各阶段的历史快照；本段为当前进度。本次不制作俄语/LaTeX/PDF报告，未启动Lab7。
'''
    for name in ['README.md', 'EXPERIMENT_SUMMARY.md', 'FINAL_CHECKLIST.md', 'REPRODUCIBILITY_GUIDE.md']:
        path = Path(name)
        old = path.read_text(encoding='utf-8')
        if '<!-- LAB6_AUDITED_STATUS -->' not in old:
            path.write_text(old.rstrip() + block, encoding='utf-8')
    readme = root / 'README.md'
    old = readme.read_text(encoding='utf-8')
    if '## Final independent regeneration' not in old:
        readme.write_text(old + '''

## Final independent regeneration

`regeneration_audit.json` confirms byte-identical regeneration of all five CSVs, both counterfactual answer files and eight PNG previews into `verification/regenerated/`, with no training. SVG creation timestamps are not compared. `reports.tools.lab6_regeneration_check` intentionally refuses to overwrite an existing regeneration directory; preserve the verification evidence and use a fresh namespace for another reproduction. `../lab5_verification/all_checkpoints_loadability.json` independently verifies all286 old checkpoints, including budget runs. Durable console logs are in `logs/`.

`requirement_status.json` separates scientific observations from procedural checks: OBS01 is FAIL because mixing never reaches20%; E108 is PASS because the specified fallback was applied faithfully. Alleta Cov=0 is disclosed rather than reclassified as good sampling. The independent audit PASS means measurements/provenance were verified, not that every scientific threshold was reached. Final audit and regeneration do not execute optimizer updates.
''', encoding='utf-8')
    print({'audit_check_count': audit['check_count'], 'handoff': 'saved'})


if __name__ == '__main__':
    main()
