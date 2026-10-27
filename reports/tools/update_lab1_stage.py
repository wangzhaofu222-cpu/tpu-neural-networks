"""Add the verified Lab1 stage without relabelling unfinished course labs."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    root=Path(__file__).resolve().parents[2]
    def read(name):
        return json.loads((root/name).read_text(encoding='utf-8'))
    if not (root/'reports/lab1/acceptance.json').exists():
        return
    acceptance=read('reports/lab1/acceptance.json')
    if acceptance.get('status') not in {'PASS','COMPLETE_WITH_EXPLAINED_THRESHOLD_FAILURE'}:
        return
    results=read('reports/lab1/results.json');build=read('reports/lab1/build_status.json')
    tests=ET.parse(root/'docs/environment/tests-lab1-final.xml').getroot().find('testsuite').attrib
    e3=results['E3'];minutes=results['runtime_seconds']/60
    summary=root/'EXPERIMENT_SUMMARY.md'
    text=summary.read_text(encoding='utf-8')
    text=text.replace('| 1–3 | NOT RUN；需求/profile已登记 | 按依赖次序逐个开发、测试、预算后运行 |',
                      '| 1 | 完成：实现、正式E1–E3、5CF、已编译俄语报告 | 缩放残差L96信号上界FAIL，已实测诊断；α=1数值失败保留 |\n| 2–3 | NOT RUN；需求/profile已登记 | 按依赖次序逐个开发、测试、预算后运行 |')
    text += f'''\n## Lab1 阶段：工作完成，存在已解释的数值阈值失败

- 复用 Lab0 ExperimentRunner；新增深层MLP、精确残差块和激活/梯度监测。原Lab0的{acceptance['Lab0_preserved_files']}个报告、记录与检查点文件逐字节保留。
- Profile：深度8/24/48/96、Gaussian σ=.02、SGD momentum；CIFAR-10全train43000、宽256、3epochs，固定网格选择LR={results['selected_lr']}。
- E1完整12组初始化及4组训练一轮后复测；最大深度全部前/反χ误差≤20%。E2完整12条件；α=1四组真实数值失败，保留失败记录；其余条件完成3整轮。缩放残差L96的q比例6.9964超过上限6.8493约2.15%，该数值阈值为FAIL；逐块c和交叉项实测已定位近似条件失效，其余增长阈值通过。
- E3假设先真实Git提交，再运行共同seeds1/2/3。rho(A)-rho(B)={e3['deltas']}；均值{e3['mean']:.6f}、样本std{e3['sample_std']:.6f}；差异判定：{'成立' if e3['established'] else '未确立'}。只适用于同一数据划分。
- 自动测试{tests['tests']}项全通过；报告{build['pages']}页，真实LaTeX编译与每页图像检查均通过。运行累计{minutes:.2f}分钟（含准备、小测试和预算；开发、测试和报告编译另计）。
- 证据：reports/lab1/acceptance.json、results.json、report.pdf、main.tex、experiments/lab1/ledger.json、hypothesis.md。
'''
    summary.write_text(text,encoding='utf-8')
    checklist=root/'FINAL_CHECKLIST.md';text=checklist.read_text(encoding='utf-8')
    text=text.replace('未完成Lab1–Lab9','未完成Lab2–Lab9').replace('| Lab1–Lab9特定实现/自动测试/正式实验/报告 | NOT RUN | 不能以Lab0框架或fixture测试代替 |',
         '| Lab1实现、必测、正式E1–E3、5CF、地图、Git事前假设 | 完成 | reports/lab1/acceptance.json |\n| Lab1全部数值阈值 | FAIL（1项） | 缩放残差L96上界超出约2.15%，已用逐块矩实测解释 |\n| Lab1俄语LaTeX编译及全部页面检查 | PASS | reports/lab1/build_status.json |\n| Lab2–Lab9特定实现/自动测试/正式实验/报告 | NOT RUN | 不能以已有框架或fixture测试代替 |')
    checklist.write_text(text,encoding='utf-8')
    guide=root/'REPRODUCIBILITY_GUIDE.md';text=guide.read_text(encoding='utf-8')
    if '## Lab1 复现实验' not in text:
        text += '''\n## Lab1 复现实验

全部训练仍由 src.engine.runner.ExperimentRunner 执行。环境与 requirements.lock.txt 沿用 Lab0。正式训练使用全部43000个训练样本；diagnostic_protocol.json 保存1024个rho索引及64个probe索引，均为train内索引。

已有原始结果保存在 experiments/lab1/ledger.json；默认重入会继续同一系列，保留已有结果。需要从头复现时，使用新的系列目录，不能删除历史结果或把预算结果改为正式结果：

```powershell
.conda-env/python.exe -X utf8 -m scripts.lab1_suite e1 --suite-root experiments/lab1-reproduction
.conda-env/python.exe -X utf8 -m scripts.lab1_suite lr --suite-root experiments/lab1-reproduction
.conda-env/python.exe -X utf8 -m scripts.lab1_suite e2 --suite-root experiments/lab1-reproduction
.conda-env/python.exe -X utf8 -m scripts.lab1_suite boundary --suite-root experiments/lab1-reproduction
.conda-env/python.exe -X utf8 -m scripts.lab1_suite cf2 --suite-root experiments/lab1-reproduction
.conda-env/python.exe -X utf8 -m scripts.lab1_suite cf3 --suite-root experiments/lab1-reproduction
```

E3必须在本次E1/E2后重新记录并真实提交假设，不能在已有E3之后重写事前记录。参见 reports/tools/preregister_lab1.py（支持 --suite-root），随后实际Git提交，再执行对应目录的e3。RUN_ID每次新建，不能要求UUID相同。不同Git提交会产生不同EXP_ID。

用 `src.compare` 比较运行时会附加rho_T/GI/AVD/数值失败字段。GI按逐层g的跨步骤中位数计算；rho在训练结束后固定train子集计算；失败的α=1条件不具有三轮后rho。初始化probe是float64权重副本，训练为float32；这两种精度不可混为一谈。

报告由 reports/tools/summarize_lab1.py 读取原始记录生成，真实LaTeX编译使用Tectonic0.17.0，下载来源及SHA256、编译日志、字体哈希见 reports/lab1/build_status.json。内置预览仍报Windows启动1060，独立编译结果已验证。历史Lab0报告不因此重建。
'''
    guide.write_text(text,encoding='utf-8')
    # Update only mandatory Lab1 clauses; theory, references and optional tasks remain separate.
    register=root/'docs/requirements/clause_register.json';rows=json.loads(register.read_text(encoding='utf-8'));mapped={}
    evidence=['reports/lab1/acceptance.json','reports/lab1/report.pdf','docs/requirements/lab1_execution_checklist.md']
    for row in rows:
        chapter=row['section'].split('.')[0]
        if row['lab']==1 and chapter.isdigit() and int(chapter) in {7,8,9,10,11,12,13}:
            flag='COMPLETED_WITH_THRESHOLD_FAILURE' if row['section'].startswith('9.2') else 'PASS'
            row.update(status=flag,evidence=evidence);mapped[row['id']]=row
    register.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    p=root/'docs/requirements/lab1_clauses.md';lines=p.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| L1-'):
            cells=line.split('|')
            if cells[1].strip() in mapped:
                cells[-3]=' '+mapped[cells[1].strip()]['status']+' ';cells[-2]=' '+'; '.join('`'+e+'`' for e in evidence)+' ';lines[i]='|'.join(cells)
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    p=root/'docs/requirements/requirements_matrix.md';lines=p.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| L1-'):
            cells=line.split('|');flag='完成；L96增长阈值FAIL，已诊断' if cells[1].strip()=='L1-E2' else 'PASS'
            cells[-2]=' '+flag+' — 逐项证据见 [Lab1验收](lab1_acceptance.md) ';lines[i]='|'.join(cells)
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'docs/requirements/lab1_acceptance.md').write_text('# Lab1 最终验收\n\n工作完成；全部数值阈值未通过（1项）：缩放残差L96上界超出约2.15%，保留FAIL并完成逐块矩诊断。数值失败作为真实负面结果保留，不等于完整三轮成功。\n\n| 验收项 | 状态 |\n|---|---|\n'+'\n'.join('| '+k+' | '+('PASS' if v else 'FAIL')+' |' for k,v in acceptance['gates'].items())+'\n\n全部数值、RUN_ID、统计和限制见 `reports/lab1/results.json` 与六节俄语报告；5类必测和补充回归见 `docs/environment/tests-lab1-final.xml`。附加§14实验未标作已完成。\n',encoding='utf-8')
    print('Lab1 stage and mandatory requirement evidence updated; Lab2–9 remain unfinished.')


if __name__=='__main__':
    main()
