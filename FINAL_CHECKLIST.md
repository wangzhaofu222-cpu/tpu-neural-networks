# 阶段验收清单

全课程状态：NOT RUN（未完成Lab2–Lab9）。Lab0：PASS。

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
| Lab2–Lab9特定实现/自动测试/正式实验/报告 | NOT RUN | 不能以已有框架或fixture测试代替 |
| Lab4 CF2 / Lab5 masks / Lab9协议与数据 | BLOCKED | 用户确认没有教师协议/专用包，missing_resources.md |
| 全课程可提交完成验收 | NOT RUN | 仅完成阶段工作，不虚报全课程完成 |
