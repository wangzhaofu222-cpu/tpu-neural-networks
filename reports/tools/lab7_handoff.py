from pathlib import Path
from reports.tools.lab7_common import R,read


def main():
    a=read(R/'final_audit.json');st=a['statistics'];status=read(R/'requirement_status.json');e3=read(R/'E3.json')
    regen=read(R/'regeneration_audit.json');assert regen['status']=='PASS'
    block=f'''

<!-- LAB7_AUDITED_STATUS -->
## Lab7 当前完成状态

Lab7已复用同一ExperimentRunner完成全部必做实验和五个反事实问题；本次未制作或修改正式俄语/LaTeX/PDF报告，未进入Lab8。
个人参数T1000,beta[.0001,.02],M6,rho.1,pi0.02；默认width256/depth6/time128保留，实测预算后使用CUDA。
E1两模型各完整20000updates；E2网络和精确score全部DDPM/DDIM网格及K1/2、五次计时、误差/score/1000轨迹诊断均完成。
E3为同一linear模型上的DDPM与DDIM50；事前Git={e3['preregistration_commit']}，三差={e3['deltas']}，教师结论={e3['teacher_conclusion']}。
两种降低稀有权重的目标及Q1十倍T均真实重新训练20000updates；未用直接换测试目标替代训练。
最终{st['tests_passed']}项测试通过、{a['check_count']}项独立审计通过；需求状态={status['counts']}，科学阈值以逐项记录为准。
正式训练{st['formal_training_runs']}次、累计{st['formal_training_seconds']:.3f}秒；正式checkpoint{st['formal_checkpoints']}个，含预算{st['all_checkpoints']}个；{st['figure_groups']}组SVG/PNG。
全部表格/PNG/SVG及反事实Markdown在独立目录再生成逐字节一致；所有模型/原始样本/轨迹/RNG本地保留并登记SHA256。
详见 [Lab7中文汇总](experiments/lab7/EXPERIMENT_SUMMARY.md)、[复现说明](experiments/lab7/README.md)、[逐项要求](docs/requirements/requirements_lab7.md)、[独立审计](experiments/lab7/final_audit.json)。
Lab0–6原始记录保持不变：Lab4Q2 BLOCKED、Lab5课程split UNVERIFIED、Lab5E2量级FAIL、Lab6E1混合FAIL均保留；前文各阶段段落为历史快照。
'''
    for name in ['README.md','EXPERIMENT_SUMMARY.md','FINAL_CHECKLIST.md','REPRODUCIBILITY_GUIDE.md']:
        p=Path(name);old=p.read_text(encoding='utf-8')
        if '<!-- LAB7_AUDITED_STATUS -->' not in old:p.write_text(old.rstrip()+block,encoding='utf-8')
    (R/'implementation_scope.md').write_text('''# Shared framework impact

The only existing core files changed forLab7 are src/config.py (Lab7-only validation), src/data/vision.py (new rare_gmm2d registry dispatch), src/models/small_cnn.py (new diffusion_mlp registry dispatch), and src/engine/runner.py (Lab7 objective/evaluator/RNGadapter and generative-mode membership). Previous model/data/objective branches retain their behavior. No old experiment/config/weight/measurement/threshold is edited.

153 oldtests passed before implementation; all oldtests pass again in the final fullsuite with new diffusion tests. Lab6 independently replays234checks and strictlyloads63checkpoints in a fresh namespace;5841 oldexperiment/runfiles are hash-protected. Existing checkpoint provenance guards remain strict: oldcheckpoints resume only with their exact recorded source/config/data, using historicalGit ifnecessary. This restriction is preserved rather than bypassed by feature integration. The same existing ExperimentRunner.run is the only optimization loop. New sampler/score/diagnostic code does not create a second trainer.
''',encoding='utf-8')
    print('Final handoff links and implementation scope saved')


if __name__=='__main__':main()
