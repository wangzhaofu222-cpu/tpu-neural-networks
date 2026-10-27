"""Generate honest phase summaries from saved acceptance, tests, and build diagnostics."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    directory=ROOT/'experiments/lab0/6116b9614f294c4c918500d92471fc4e'
    ledger=read(directory/'ledger.json'); report=ROOT/'reports/lab0'
    acceptance=read(report/'acceptance.json') if (report/'acceptance.json').exists() else {'status':'NOT RUN'}
    build=read(report/'build_status.json'); compatibility=ROOT/'docs/environment/release-compatibility.json'
    code_ok=read(compatibility)['status']=='PASS' if compatibility.exists() else False
    lab0_ok=acceptance['status']=='PASS' and build['visual_QA_status']=='PASS' and code_ok
    test=ROOT/'docs/environment/tests-lab0-release.xml'
    if not test.exists(): test=ROOT/'docs/environment/tests-lab0-final.xml'
    stats=ET.parse(test).getroot().find('testsuite').attrib
    passed=int(stats['tests'])-sum(int(stats[x]) for x in ['errors','failures','skipped'])
    e2=ledger['checks']['E2']; extra=read(directory/'CF04-extra.json') if (directory/'CF04-extra.json').exists() else {'runs':[]}
    e0=read(ROOT/'docs/environment/e0-selected.json'); selected=read(ROOT/'runs'/e0['RUN_ID']/'summary.json')
    status='PASS' if lab0_ok else 'NOT RUN（最终代码/报告验收待补齐）'
    summary=f'''# 实验项目阶段汇总

Phase A/B已完成；Lab0阶段验收：**{status}**。整个Lab0–Lab9课程项目尚未完成，不能作为全课程完成提交。

## 已完成

- 10份教师PDF、152页全文审计、2153个原文条款ID；原PDF哈希不变。2153包含理论和参考文献，不等于2153项必做实验。
- 项目独立Python3.11.17 / PyTorch2.7.1+cu126 / torchvision0.22.1+cu126，精确锁定依赖；CPU与RTX3060 Laptop CUDA实测可用。
- 公共runner、五CLI、哈希profile、数据角色隔离、完整checkpoint/RNG/GradScaler、逐step日志、独立evaluate、比较与性能测量。
- 自动测试{passed}/{stats['tests']}通过，errors={stats['errors']}、failures={stats['failures']}、skipped={stats['skipped']}，{float(stats['time']):.3f}秒。证据：`{test.relative_to(ROOT).as_posix()}`。合成fixture测试仅验证实现。

## Lab0真实结果

- 真实CPU预算24.09秒/epoch，12epoch；13个逻辑训练条件估计62.63分钟。主serial-suite实际{ledger['wall_seconds']/60:.2f}分钟；此前E0/失败尝试、后续CF4补充与注册profiling另计。
- E0 loss 1.964259→{selected['train_loss']:.6f}，val accuracy={selected['validation_metric']:.4f}，独立evaluate完全吻合。首次K10的loss比值0.500332未通过，保留FAIL记录；K12从头重训通过。
- A/CPU、B/CUDA、C/CUDA+AMP：E1逐step loss与最终指标重复一致；E3完整进程重启恢复后参数相对差均为0。有效optimizer steps A/B为2016，C为2015；C实际有1次overflow跳步。
- E2预定5seed：mean={e2['mean']:.6f}，样本std={e2['sample_std']:.6f}，95% Student t CI=[{e2['ci95'][0]:.6f},{e2['ci95'][1]:.6f}]；2std参考效应={e2['minimum_effect_heuristic']:.6f}。
- workers2配对与workers0基准的全部2016个step losses完全相同。只恢复权重的最终参数相对差={ledger['checks']['CF03']['delta_theta']:.6f}，不能当成正确恢复。
- CF4额外{len(extra['runs'])}个seed单独登记，不重算预定5seed统计；数据1字节干预和optimizer/lr/RNG诊断均有真实记录。
- E4主CPU/GPU系列各50预热/100测量，全部B的CV≤10%；附加GPU N300诊断在B1/B16得到CV约20.8%/13.6%，阈值状态FAIL，已保留并在报告披露，不能把主系列通过扩展为普遍延迟稳定保证。
- 俄语六节报告：`reports/lab0/report.pdf`、`main.tex`、`report.ru.md`；图表、RUN_ID注册表、实验验收及构建状态同目录。
- 主E1/E2/E3与补充训练固定提交`{ledger['code_commit']}`。发布时显式split参数及比较显示的改动另有兼容验证；不改写历史manifest。

## 课程状态

| Lab | 实现/正式实验/报告 | 限制 |
|---|---|---|
| 0 | {status} | PDF经渲染检查；LaTeX编译器启动1060，tex编译未验证 |
| 1–3 | NOT RUN；需求/profile已登记 | 按依赖次序逐个开发、测试、预算后运行 |
| 4 | NOT RUN；CF2完整验收BLOCKED | 缺课程协议§4.6颜色控制集 |
| 5 | NOT RUN；正式masks核验BLOCKED | 缺课程Planetoid masks |
| 6–7 | NOT RUN；需求/profile已登记 | 长链、训练、采样与CF须各自先做真实预算 |
| 8 | NOT RUN | 缺Lab2训练checkpoint；近OOD附加缺11类名单 |
| 9 | BLOCKED（正式数据） | 缺人脸帧/音频预处理包、演员/clip划分；编码器已按用户授权选cnn_lab2 |

## 已知边界

课程数据协议缺失，自建split不声称逐索引等同教师文件。CIFAR官方页未列独立许可证，不分发原数据。CPU温度未测，用户要求不等待传感器。断点恢复保证仅限完整epoch边界和记录的代码/环境。没有遗留后台或通宵训练计划。LaTeX编译未验证与PDF渲染验收分开记录。
'''
    (ROOT/'EXPERIMENT_SUMMARY.md').write_text(summary,encoding='utf-8')
    checklist=f'''# 阶段验收清单

全课程状态：NOT RUN（未完成Lab1–Lab9）。Lab0：{status}。

| 要求 | 状态 | 证据 |
|---|---|---|
| 10份原PDF/hash、152页登记、需求矩阵 | PASS | docs/requirements/source_inventory.json、clause_register.json、verify_audit.py |
| Python≥3.11/PyTorch2、CPU/CUDA、精确依赖锁 | PASS | docs/environment/current.json、requirements.lock.txt |
| Profile生成与Lab9冲突处理 | PASS | configs/profiles/current.json；用户授权来源优先，不是教师确认 |
| Lab0十一项必测及补充边界测试 | {'PASS' if passed==int(stats['tests']) else 'FAIL'} | {test.relative_to(ROOT).as_posix()}、test_traceability.md |
| Lab0真实数据预算、正式E0–E4、8CF | {acceptance['status']} | reports/lab0/acceptance.json、report_evidence.json、原始ledger/runs |
| Lab0数据split显式配置及发布兼容 | {'PASS' if code_ok else 'NOT RUN'} | docs/environment/release-compatibility.json |
| 附加GPU N300的CV≤10% | FAIL | B1约20.8%、B16约13.6%；报告中保留原始波动及局限，不影响主N100已通过的事实 |
| Lab0俄语六节报告/PDF视觉验收 | {'PASS' if build['visual_QA_status']=='PASS' else 'NOT RUN'} | reports/lab0/report.pdf、build_status.json |
| Lab0 LaTeX编译 | BLOCKED | 预览进程CreateProcessWithLogonW启动错误1060；tex保留，PDF独立导出 |
| Lab1–Lab9特定实现/自动测试/正式实验/报告 | NOT RUN | 不能以Lab0框架或fixture测试代替 |
| Lab4 CF2 / Lab5 masks / Lab9协议与数据 | BLOCKED | 用户确认没有教师协议/专用包，missing_resources.md |
| 全课程可提交完成验收 | NOT RUN | 仅完成阶段工作，不虚报全课程完成 |
'''
    (ROOT/'FINAL_CHECKLIST.md').write_text(checklist,encoding='utf-8')
    # Preserve any later, independently verified stage when refreshing Lab0 status.
    import runpy
    later=ROOT/'reports/tools/update_lab1_stage.py'
    if later.exists():
        runpy.run_path(str(later),run_name='__main__')
    print('Lab0 stage:',status)


if __name__=='__main__': main()
