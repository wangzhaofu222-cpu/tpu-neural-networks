# 验收清单（工作版本，不是最终课程验收）

| 要求 | 状态 | 证据/缺失 |
|---|---|---|
| 原PDF只读/hash与全152页登记 | PASS | docs/requirements/source_inventory.json、verify_audit.py |
| 五份需求文档与全课程profile规则 | PASS | docs/requirements/、configs/profiles/current.json |
| Python≥3.11 / PyTorch2 / CPU+CUDA | PASS | docs/environment/current.json |
| 全依赖精确锁与无冲突 | PASS | requirements.lock.txt、conda-explicit-win64.txt、pip check |
| Lab0 §25十一项基础自动测试 | PASS | 最新21/21 tests；docs/environment/test_traceability.md |
| 新进程完整epoch边界resume | PASS（fixture范围） | test_resume_in_new_process_exact_trajectory |
| 真实CIFAR单epoch预算 | PASS（测量完成） | docs/environment/lab0_budget.json |
| 至少13次CPU/worker完整训练+CUDA附加 | NOT RUN | 没有正式结果；测试不能替代 |
| E0阈值/5seed统计/真实E3/E4验收 | NOT RUN | 待全部正式日志 |
| Lab0八个CF实测答案与六节报告 | NOT RUN | 不生成虚构数字或RUN_ID |
| Lab1–9实现/特定测试/正式实验/报告 | NOT RUN | 仅需求与profile已完成 |
| Lab4 CF2 / Lab5 masks / Lab9数据协议 | BLOCKED | 用户确认无课程数据协议和专用包 |
| Lab9编码器冲突 | PASS（项目决定已记录） | 两原始字段保留，用户授权visual_encoder来源优先；不是教师确认 |
| 全课程最终可提交验收 | NOT RUN | 目前不可提交为已完成课程 |
