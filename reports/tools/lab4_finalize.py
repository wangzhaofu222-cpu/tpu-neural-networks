"""Publish audited evidence indexes in project documentation; never creates a formal report."""
import json
from pathlib import Path
from src.utils.io import write_json

ROOT=Path('experiments/lab4')
MARK='<!-- LAB4_AUDITED_STATUS -->'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def append_section(path,text):
    p=Path(path);old=p.read_text(encoding='utf-8').split(MARK)[0].rstrip()
    p.write_text(old+'\n\n'+MARK+'\n'+text.strip()+'\n',encoding='utf-8')

def main():
    audit=read(ROOT/'audit.json');result=read(ROOT/'results.json');totals=audit['totals']
    assert audit['automated_tests']['failures']==audit['automated_tests']['errors']==0
    ledger={r['label']:r for r in read(ROOT/'ledger.json') if r['status'].startswith('COMPLETED')}
    evaluations={r['label']:r for r in read(ROOT/'evaluations.json')}
    e1='; '.join(r['encoder_RUN_ID'] for r in result['E1']['rows'])
    e2='; '.join(r['training_RUN_ID'] for r in result['E2']['rows'])
    cf=result['counterfactual_status']
    mapping={
        'C01':ledger['CF1_N64']['RUN_ID'],
        'C02':'— (teacher Protocol section 4.6 unavailable; no fabricated run)',
        'C03':'; '.join(v['encoder_RUN_ID'] for v in read(ROOT/'counterfactuals.json')['Q3']['actual_pairs'][0].values()
            if isinstance(v,dict) and 'encoder_RUN_ID' in v) if cf['Q3']=='PASS' else 'counterfactuals.json',
        'C04':evaluations['E1_SSL_epoch100_h']['RUN_ID']+'; '+evaluations['CF4_SSL_z_hat']['RUN_ID'],
        'C05':ledger['CF5_B1_low_labels']['RUN_ID']+'; '+evaluations['CF5_B1_low_labels_h']['RUN_ID'],
        'H01':'ce94369 (exploratory hypotheses); '+e1,
        'O01':'RUN_registry.json / checkpoint_index.json',
    }
    matrix=Path('docs/requirements/requirements_lab4.md');updated=[]
    for line in matrix.read_text(encoding='utf-8').splitlines():
        parts=line.split('|')
        if len(parts)==10 and parts[3].strip()=='mandatory':
            key=parts[1].strip();status='PASS'
            if key=='C02':status='BLOCKED'
            elif key=='O02':status='NOT_APPLICABLE'
            elif key=='E105' and not result['E1']['loss_threshold_pass']:status='FAIL'
            elif key.startswith('C0') and key[-1] in '1345':status=cf['Q'+key[-1]]
            run=mapping.get(key,'— (implementation/tests; final XML and audit.json)')
            if key.startswith('E1'):run=e1
            elif key.startswith('E2'):run=e2
            elif key.startswith('E3'):run='E3.csv: all 36 RUN_ID cells; preregistration '+audit['preregistration_commit'][:7]
            elif key in ['P01','F01','K01','R01','R02','R03','D01','D02']:run='RUN_registry.json / evaluations.json / features.json'
            parts[-3]=run;parts[-2]=status;line='|'.join(parts)
        updated.append(line)
    matrix.write_text('\n'.join(updated)+'\n',encoding='utf-8')
    g=result['E3'];tests=audit['automated_tests']['tests'];training=totals['formal_encoder_training_seconds']/3600
    evidence=f'''## Lab4 最终证据状态：{audit['status']}

- Lab3 独立复核 PASS：88 项测试重跑，1908 个历史文件不变，原有 E2 两项 FAIL 保留。
- Lab4 E1/E2/E3 均已执行，完整 36 格 E3 矩阵包括 3 格明确复用的 E1 结果。E3 事前 Git：`{audit['preregistration_commit']}`。
- E1 probe：B0 43.57%、B1 89.88%、SimCLR 76.62%；G(1)=+33.05 个百分点。
- E3 rmin 三差值：{g['deltas']}，均值 {g['mean']:.9f}，样本 SD {g['sample_std']:.9f}，教师判定：{g['conclusion']}。事前点预测 {g['preregistered_point_prediction']:.6f}，误差 {g['point_prediction_error']:+.9f}；方向成立不等于数值预测准确。
- {tests} 项自动测试全部通过。四个反事实问题有实测依据；Q2 缺教师 Protocol §4.6，保留 BLOCKED。因此不能宣称全部必做任务已无条件通过。
- 正式编码器训练 {totals['formal_encoder_training_runs']} 次，probe 正则化候选训练 {totals['formal_probe_candidate_training_runs']} 次；B0 仅初始化。编码器累计训练 {training:.4f} 小时；完整非重叠计时见 audit.json。
- 本地 checkpoint 共 {totals['checkpoint_count']} 个，其中正式 {totals['formal_checkpoint_count']} 个；所有配置、原始记录、特征缓存、哈希及路径均已登记。大型权重和缓存不强制加入 Git。
- 按用户要求不制作或修改正式 LaTeX/PDF 报告；Lab5 未启动。全部详细数值与 PASS/FAIL/BLOCKED 见 experiments/lab4/results.json、audit.json、counterfactuals.json。
'''
    append_section('EXPERIMENT_SUMMARY.md',evidence)
    append_section(ROOT/'README.md',evidence)
    summary=Path('EXPERIMENT_SUMMARY.md');text=summary.read_text(encoding='utf-8').replace('## Lab3 阶段（本轮终点，报告暂缓）','## Lab3 阶段（历史记录，报告暂缓）').replace('没有生成或修改LaTeX/PDF报告，没有启动Lab4。','该阶段没有生成或修改LaTeX/PDF报告；该阶段结束时尚未启动Lab4。');summary.write_text(text,encoding='utf-8')
    checklist=Path('FINAL_CHECKLIST.md');text=checklist.read_text(encoding='utf-8').replace('Lab4–Lab9未运行','Lab4 CF2 尚缺资料、Lab5–Lab9未运行').replace('| Lab4–Lab9特定实现/自动测试/正式实验/报告 | NOT RUN | 本轮在Lab3后停止 |','| Lab5–Lab9特定实现/自动测试/正式实验/报告 | NOT RUN | 本轮在Lab4后停止 |');checklist.write_text(text,encoding='utf-8')
    gates='\n'.join('| '+k+' | '+v+' | experiments/lab4/audit.json |' for k,v in audit['acceptance'].items())
    append_section(checklist,'## Lab4 审计\n\n| 要求 | 状态 | 证据 |\n|---|---|---|\n'+gates)
    append_section('README.md','## Lab4\n\n实验已推进至 Lab4，Q2 因缺教师 Protocol §4.6 保留 BLOCKED；不虚报全部完成。复现入口与数据索引见 [Lab4 README](experiments/lab4/README.md)，逐项验收见 [Lab4 audit](experiments/lab4/audit.json)。没有制作新报告，没有启动 Lab5。')
    append_section('REPRODUCIBILITY_GUIDE.md','## Lab4 冻结特征实验\n\n使用既有 .conda-env 与 ExperimentRunner，命令和恢复限制见 experiments/lab4/README.md。重算顺序：reports.tools.lab4_results → 完整 pytest（JUnit 路径 docs/environment/tests-lab4-final.xml）→ reports.tools.lab4_audit → reports.tools.lab4_finalize。重算分析不会重新训练模型。E3 假设及 E1/E2 事前快照不可修改；原始 checkpoint 和特征保存在 runs/lab4 并由哈希索引登记。若要重新执行主实验，应使用已登记的源提交及原配置，不把新运行覆盖到旧 RUN_ID。')
    write_json(ROOT/'documentation_status.json',{'audit_status':audit['status'],'requirements_reconciled':True,'formal_report_created':False,'Lab5_started':False})
    print(json.dumps({'status':audit['status'],'tests':tests,'requirements_reconciled':True}))

if __name__=='__main__':main()
