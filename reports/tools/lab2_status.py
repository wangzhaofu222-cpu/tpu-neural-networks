"""Update current stage summaries without regenerating historical reports."""
import json
from pathlib import Path
def main():
    r=json.loads(Path('experiments/lab2/results.json').read_text(encoding='utf-8'))
    established=r['E3']['established'];tests=r['automated_tests']['tests']
    hours=r.get('training_seconds_lower_bound_including_interrupted_epochs',r['total_training_seconds'])/3600
    text=(f'\n\n## Lab2 阶段（报告按用户要求暂缓）\n\n'
          f'- 复用公共 ExperimentRunner，完成标准 ResNet18/ViT、E1 五比例、E2 三扰动、E3 三配对 seed、地图及全部五个反事实问题。\n'
          f'- 完成 {r["formal_training_executions"]} 个正式训练运行，逻辑轨迹共 {r["formal_optimizer_updates"]} 次优化器更新；已记录训练耗时下界 {hours:.3f} 小时。复用记录不重复计数。\n'
          '- 曾有1个外部中断父进程，从完整checkpoint精确恢复；499个已记录更新被重放。20次正式进程调用中19次完成，记录的实际更新含重放为181005次。中断父进程精确总时长不可恢复，不能伪报精确累计时长。\n'
          f'- 自动测试 {tests} 项通过；E3 平均差 {r["E3"]["mean"]:.6f}，样本标准差 {r["E3"]["sample_std"]:.6f}，差异判定：{"成立" if established else "未建立"}。\n'
          '- 完整 PASS/FAIL/边界：experiments/lab2/results.json；原始 ledger、独立评估 RUN_ID、图表、反事实和 checkpoint 索引同目录。\n'
          '- 已保留全量模型，解除 Lab8 对 Lab2 checkpoint 的依赖；Lab8/9 尚未启动，其他数据缺口仍保留。未生成新报告，未启动 Lab3。\n')
    p=Path('EXPERIMENT_SUMMARY.md');s=p.read_text(encoding='utf-8')
    s=s.replace('| 2–3 | NOT RUN；需求/profile已登记 | 按依赖次序逐个开发、测试、预算后运行 |',
                '| 2 | 正式实验完成；新报告按用户要求暂缓 | 逐项状态见 experiments/lab2/results.json |\n| 3 | NOT RUN | 本轮在Lab2结束后停止 |')
    s=s.replace('缺Lab2训练checkpoint；近OOD附加缺11类名单','Lab2 checkpoint已保存；近OOD附加缺11类名单')
    if '## Lab2 阶段（报告按用户要求暂缓）' not in s:s+=text
    p.write_text(s,encoding='utf-8')
    p=Path('FINAL_CHECKLIST.md');s=p.read_text(encoding='utf-8').replace('未完成Lab2–Lab9','未完成Lab3–Lab9')
    s=s.replace('| Lab2–Lab9特定实现/自动测试/正式实验/报告 | NOT RUN | 不能以已有框架或fixture测试代替 |',
                '| Lab2实现/正式E1–E3/5CF/地图/checkpoint | 完成 | experiments/lab2/results.json；逐项数值状态见其中acceptance |\n| Lab2新报告 | 按用户要求暂缓 | 不生成LaTeX/PDF |\n| Lab3–Lab9特定实现/自动测试/正式实验/报告 | NOT RUN | 本轮没有启动Lab3 |')
    p.write_text(s,encoding='utf-8')
    p=Path('README.md');s=p.read_text(encoding='utf-8').replace('Lab2–9尚未实现/训练/完成报告；',
       'Lab2已完成标准模型、正式实验、多seed统计、反事实及checkpoint保存，新报告按用户要求暂缓，证据见`experiments/lab2/README.md`。Lab3–9尚未实现/训练/完成报告；')
    p.write_text(s,encoding='utf-8')
    p=Path('docs/requirements/requirements_matrix.md');lines=p.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| L2-'):
            status='按用户要求暂缓新报告' if line.startswith('| L2-REP') else 'PASS — 逐项证据见 experiments/lab2/results.json / lab2_execution_checklist.md'
            if line.startswith('| L2-E3') and not established:status='完成；统计差异未建立（FAIL）— experiments/lab2/results.json'
            fields=line.split('|');fields[-2]=' '+status+' ';lines[i]='|'.join(fields)
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
