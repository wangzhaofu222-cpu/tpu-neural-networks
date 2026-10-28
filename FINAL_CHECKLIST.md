# 阶段验收清单

全课程状态：尚未完成（Lab4–Lab9未运行；Lab2/3新报告按要求暂缓）。Lab0：PASS。

| 要求 | 状态 | 证据 |
|---|---|---|
| 10份原PDF/hash、152页登记、需求矩阵 | PASS | docs/requirements/source_inventory.json、clause_register.json、verify_audit.py |
| Python≥3.11/PyTorch2、CPU/CUDA、精确依赖锁 | PASS | docs/environment/current.json、requirements.lock.txt |
| Profile生成与Lab9冲突处理 | PASS | configs/profiles/current.json；用户授权来源优先，不是教师确认 |
| Lab0十一项必测及补充边界测试 | PASS | docs/environment/tests-lab0-release.xml、test_traceability.md |
| Lab0真实数据预算、正式E0–E4、8CF | PASS | reports/lab0/acceptance.json、report_evidence.json、原始ledger/runs |
| Lab0数据split显式配置及发布兼容 | PASS | docs/environment/release-compatibility.json |
| 附加GPU N300的CV≤10% | FAIL | B1约20.8%、B16约13.6%；报告中保留原始波动及局限，不影响主N100已通过的事实 |
| Lab0俄语六节报告/PDF视觉验收 | PASS | reports/lab0/report.pdf、build_status.json |
| Lab0 LaTeX编译 | BLOCKED | 预览进程CreateProcessWithLogonW启动错误1060；tex保留，PDF独立导出 |
| Lab1实现、必测、正式E1–E3、5CF、地图、Git事前假设 | 完成 | reports/lab1/acceptance.json |
| Lab1全部数值阈值 | FAIL（1项） | 缩放残差L96上界超出约2.15%，已用逐块矩实测解释 |
| Lab1俄语LaTeX编译及全部页面检查 | PASS | reports/lab1/build_status.json |
| Lab2实现/正式E1–E3/5CF/地图/checkpoint | 完成 | experiments/lab2/results.json；逐项数值状态见其中acceptance |
| Lab2新报告 | 按用户要求暂缓 | 不生成LaTeX/PDF |
| Lab2独立完整性复核 | PASS | experiments/lab2_verification/audit.json；70测试、38checkpoint、原记录不变 |
| Lab3必做实现/E1–E3/注意力诊断/5CF/事前假设 | PASS | experiments/lab3/results.json、audit.json；4d6e559事前提交 |
| Lab3 E1两项验证阈值 | PASS | none=.057861；位置编码增益=.403381 |
| Lab3 E2原始显存渐近斜率区间 | FAIL | 两后端原始FAIL保留；未达渐近区间 |
| Lab3 E2按PDF§9.2替代分解 | PASS | E2_analysis.json；实际峰值、二次/线性/常数分解分开记录 |
| Flash Attention可用性（不是必选backend） | BLOCKED | 此Windows构建未编译；forced-efficient已完成第二后端实测 |
| Lab3 E3三seed差异判定 | PASS | mean−.072113、sampleSD=.015833，三差同负且absmean>=2SD |
| Lab0–Lab3全部自动测试 | PASS | docs/environment/tests-lab3-final.xml；88通过，0失败/跳过 |
| Lab3新报告 | NOT_APPLICABLE | 用户明确本轮不制作报告 |
| Lab4–Lab9特定实现/自动测试/正式实验/报告 | NOT RUN | 本轮在Lab3后停止 |
| Lab4 CF2 / Lab5 masks / Lab9协议与数据 | BLOCKED | 用户确认没有教师协议/专用包，missing_resources.md |
| 全课程可提交完成验收 | NOT RUN | 仅完成阶段工作，不虚报全课程完成 |
