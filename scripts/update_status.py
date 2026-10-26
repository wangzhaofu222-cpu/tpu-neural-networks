"""Produce evidence-based stage summaries without inventing any experimental results."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

root=Path(__file__).resolve().parents[1]
env=root/'docs/environment'
test_file=env/'tests-lab0-final.xml'
if not test_file.exists():
    test_file=env/'tests-lab0.xml'
suite=ET.parse(test_file).getroot().find('testsuite')
stats=suite.attrib
passed=int(stats['tests'])-int(stats['failures'])-int(stats['errors'])-int(stats['skipped'])
ok=not int(stats['failures']) and not int(stats['errors'])
profile=json.loads((root/'configs/profiles/current.json').read_text())
budget=json.loads((env/'lab0_budget.json').read_text()) if (env/'lab0_budget.json').exists() else None
budget_text=(f"真实 CPU 单epoch {budget['epoch_seconds']:.2f} 秒；13×{budget['epochs_planned']}epoch估计 "
             f"{budget['mandatory_13_runs_estimated_seconds']/3600:.2f}小时，不含CUDA附加实验与诊断。"
             f"教师60秒epoch目标：{'达到' if budget['teacher_epoch_target_met'] else '未达到，停止扩大训练规模'}。"
             f"预算RUN_ID `{budget['RUN_ID']}`，不是已完成E0。") if budget else '尚未完成真实数据单epoch预算，禁止启动完整训练系列。'
summary=f'''# 实验项目阶段汇总

这是Phase A/B与Lab0开发阶段记录，不是课程完成声明。

## 已完成的实证工作

- 10 PDF/152页全文审计、2153个原文条款ID；原PDF哈希不变校验PASS。
- 五份需求文档、逐Lab原文登记、可追踪主矩阵、数据依赖/缺失/歧义登记。
- 独立Conda Python3.11.17 / torch2.7.1+cu126 / torchvision0.22.1+cu126；依赖检查通过；CPU/CUDA真实运算检查通过。
- Profile `{profile['profile']['id']}`：train_fraction={profile['data']['train_fraction']}、width={profile['model']['width']}、depth={profile['model']['depth']}、batch={profile['train']['batch_size']}、{profile['optimizer']['name']}、{profile['scheduler']['name']}。未保存原学号。
- 公共runner/config/manifest/完整epoch边界checkpoint/JSONL/compare/独立evaluate/profiling已实现。
- 最新测试：{passed}/{stats['tests']}通过，failures={stats['failures']}、errors={stats['errors']}、skipped={stats['skipped']}，{float(stats['time']):.3f}秒；证据`docs/environment/tests-lab0.xml`。
- 测试真实验证新进程resume参数相对差为0、逐step loss一致，CPU/CUDA均运行；全部训练测试是明确标记的fixture，不能算正式实验。
- CPU与CUDA前向性能数据见`docs/environment/profile-*-development.json`，各B≥10预热/30测量；有独立RUN_ID。它们是开发诊断，不是完整正式E4验收。

## 真实训练预算

{budget_text}

## 课程验收状态

| Lab | 实现 | 正式实验 | 俄语报告 | 当前限制 |
|---|---|---|---|---|
| 0 | 公共框架与基础测试PASS；完整实验编排/CF/自动报告尚未完成 | NOT RUN | NOT RUN | 预算与数据协议；未完成至少13次完整训练 |
| 1–3 | NOT RUN；profile/需求已登记 | NOT RUN | NOT RUN | 先通过Lab0 |
| 4 | NOT RUN；profile/需求已登记 | BLOCKED（完整验收CF2） | NOT RUN | 缺课程协议§4.6颜色控制集；主实验仍需上游完成 |
| 5 | NOT RUN | BLOCKED（正式mask核验） | NOT RUN | 缺课程Planetoid masks |
| 6–7 | NOT RUN；profile/需求已登记 | NOT RUN | NOT RUN | 先通过Lab0，长链/采样/CF需另预算 |
| 8 | NOT RUN | NOT RUN | NOT RUN | 尚无Lab2真实checkpoint；nearOOD附加缺11类名单 |
| 9 | NOT RUN | BLOCKED | NOT RUN | 缺RAVDESS课程预处理包/演员划分；encoder冲突已按用户授权处理 |

## 仍需明确的事实

课程协议不存在；不将自建split声称为教师标准逐索引split。CIFAR官方页面未列独立许可证，已记录，不声称完成全部许可审查。CPU温度未测，用户已要求忽略等待项；预算/线程限制仍保留。没有启动无人值守通宵任务。

后续：真实CPU单epoch预算门→补齐Lab0正式编排/CF与报告工具→按可行预算正式实验并验收；Lab0通过后逐个Lab推进。受缺失资源影响的部分等待教师材料，其余不虚报完成。
'''
(root/'EXPERIMENT_SUMMARY.md').write_text(summary,encoding='utf-8')
checklist=f'''# 验收清单（工作版本，不是最终课程验收）

| 要求 | 状态 | 证据/缺失 |
|---|---|---|
| 原PDF只读/hash与全152页登记 | PASS | docs/requirements/source_inventory.json、verify_audit.py |
| 五份需求文档与全课程profile规则 | PASS | docs/requirements/、configs/profiles/current.json |
| Python≥3.11 / PyTorch2 / CPU+CUDA | PASS | docs/environment/current.json |
| 全依赖精确锁与无冲突 | PASS | requirements.lock.txt、conda-explicit-win64.txt、pip check |
| Lab0 §25十一项基础自动测试 | {'PASS' if ok else 'FAIL'} | 最新{passed}/{stats['tests']} tests；docs/environment/test_traceability.md |
| 新进程完整epoch边界resume | {'PASS（fixture范围）' if ok else 'FAIL'} | test_resume_in_new_process_exact_trajectory |
| 真实CIFAR单epoch预算 | {'PASS（测量完成）' if budget else 'NOT RUN'} | {'docs/environment/lab0_budget.json' if budget else '等待官方数据下载/计时'} |
| 至少13次CPU/worker完整训练+CUDA附加 | NOT RUN | 没有正式结果；测试不能替代 |
| E0阈值/5seed统计/真实E3/E4验收 | NOT RUN | 待全部正式日志 |
| Lab0八个CF实测答案与六节报告 | NOT RUN | 不生成虚构数字或RUN_ID |
| Lab1–9实现/特定测试/正式实验/报告 | NOT RUN | 仅需求与profile已完成 |
| Lab4 CF2 / Lab5 masks / Lab9数据协议 | BLOCKED | 用户确认无课程数据协议和专用包 |
| Lab9编码器冲突 | PASS（项目决定已记录） | 两原始字段保留，用户授权visual_encoder来源优先；不是教师确认 |
| 全课程最终可提交验收 | NOT RUN | 目前不可提交为已完成课程 |
'''
(root/'FINAL_CHECKLIST.md').write_text(checklist,encoding='utf-8')
print(f'Stage summaries updated from {stats["tests"]} tests and '+('real budget' if budget else 'no real budget yet'))
