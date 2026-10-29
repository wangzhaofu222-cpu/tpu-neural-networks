# 阶段验收清单

全课程状态：尚未完成（Lab4 CF2 尚缺资料、Lab5–Lab9未运行；Lab2/3新报告按要求暂缓）。Lab0：PASS。

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
| Lab5–Lab9特定实现/自动测试/正式实验/报告 | NOT RUN | 本轮在Lab4后停止 |
| Lab4 CF2 / Lab5 masks / Lab9协议与数据 | BLOCKED | 用户确认没有教师协议/专用包，missing_resources.md |
| 全课程可提交完成验收 | NOT RUN | 仅完成阶段工作，不虚报全课程完成 |

<!-- LAB4_AUDITED_STATUS -->
## Lab4 审计

| 要求 | 状态 | 证据 |
|---|---|---|
| historical_Lab0_to_Lab3_unchanged | PASS | experiments/lab4/audit.json |
| Lab3_independent_audit | PASS | experiments/lab4/audit.json |
| mandatory_tests_and_regressions | PASS | experiments/lab4/audit.json |
| N256_global_NT_Xent_not_microbatch_approximation | PASS | experiments/lab4/audit.json |
| GPU100_25_epoch_protocol | PASS | experiments/lab4/audit.json |
| trainonly_unlabelled_pretraining_cal_guard | PASS | experiments/lab4/audit.json |
| E1_three_encoders_common_probe_knn | PASS | experiments/lab4/audit.json |
| E1_five_SSL_checkpoints_including0 | PASS | experiments/lab4/audit.json |
| E1_E2_exploratory_hypotheses_precede_formal_training | PASS | experiments/lab4/audit.json |
| E1_final_loss_at_or_below_random | PASS | experiments/lab4/audit.json |
| E1_G1_sign_reported | PASS | experiments/lab4/audit.json |
| E2_six_variants_only_augmentation_changes | PASS | experiments/lab4/audit.json |
| E2_conditional_thresholds_applied | PASS | experiments/lab4/audit.json |
| E3_preregistered_before_new_runs | PASS | experiments/lab4/audit.json |
| E3_complete36_matrix_nested_shared_indices | PASS | experiments/lab4/audit.json |
| E3_teacher_rule_applied | PASS | experiments/lab4/audit.json |
| frozen_encoder_full_state_and_feature_repeatability | PASS | experiments/lab4/audit.json |
| representation_scale_rank_spectrum_raw_recomputed | PASS | experiments/lab4/audit.json |
| CF1 | PASS | experiments/lab4/audit.json |
| CF2 | BLOCKED | experiments/lab4/audit.json |
| CF3 | PASS | experiments/lab4/audit.json |
| eight_figures_visually_verified_and_hashed | PASS | experiments/lab4/audit.json |
| CF4 | PASS | experiments/lab4/audit.json |
| CF5 | PASS | experiments/lab4/audit.json |
| formal_report | NOT_APPLICABLE | experiments/lab4/audit.json |
| optional_section14 | NOT_APPLICABLE | experiments/lab4/audit.json |


<!-- LAB5_AUDITED_STATUS -->
## Lab5 图结构实验

Lab5 已实施 E1–E3、35次完整模型训练和102次逐层 probe，130项自动测试通过。详见 [Lab5 实验汇总](experiments/lab5/EXPERIMENT_SUMMARY.md)、[复现说明](experiments/lab5/README.md) 和 [独立审计](experiments/lab5/final_audit.json)。正式运行均为200epochs；预算测试单独保存。E2事前记录为25b13ff，量级预测失败保留，不改写假设。PyG public masks已获用户授权，但教师独立split仍UNVERIFIED；Lab4颜色协议Q2仍BLOCKED。未制作正式报告，未执行Lab6。

<!-- LAB6_AUDITED_STATUS -->
## Lab6 当前完成状态

Lab6 已完成全部必做实验，复用既有 ExperimentRunner；11次正式模型训练各5000次更新。
153项自动测试通过，独立审计234项通过，5份CSV、两种反事实答案和8张PNG独立再生成逐字节一致。
当前需求状态为57 PASS、1 FAIL、0 BLOCKED、4 NOT_APPLICABLE。FAIL为E1实测中心间混合不足20%；已按教师后备规则选择eta=.02，未掩盖全部eta的Cov=0。
E2按val配对SEM选择Kf=8；E3三seed的MMD²差异满足教师统计判据。事前Git为214ea6627908fcbdf9d168b02b3bf59760375003。
完整中文结果、RUN_ID和表格见 [Lab6汇总](experiments/lab6/EXPERIMENT_SUMMARY.md)，
[逐项要求](docs/requirements/requirements_lab6.md)，[独立审计](experiments/lab6/final_audit.json)，
[复现说明](experiments/lab6/README.md)。正式训练累计1952.580秒；55个正式checkpoint，含预算共63个，本地保留且有哈希索引。
Lab5旧130测试、625审计及全部286个checkpoint加载复核通过，5450个历史文件未改；Lab5 E2量级预测FAIL、课程split UNVERIFIED、Lab4 Q2 BLOCKED均保留。
这些段落是各阶段的历史快照；本段为当前进度。本次不制作俄语/LaTeX/PDF报告，未启动Lab7。

<!-- LAB7_AUDITED_STATUS -->
## Lab7 当前完成状态

Lab7已复用同一ExperimentRunner完成全部必做实验和五个反事实问题；本次未制作或修改正式俄语/LaTeX/PDF报告，未进入Lab8。
个人参数T1000,beta[.0001,.02],M6,rho.1,pi0.02；默认width256/depth6/time128保留，实测预算后使用CUDA。
E1两模型各完整20000updates；E2网络和精确score全部DDPM/DDIM网格及K1/2、五次计时、误差/score/1000轨迹诊断均完成。
E3为同一linear模型上的DDPM与DDIM50；事前Git=f65a5ff1e4eb61ce9c387c5a3c75926beaca0f55，三差=[-0.036199999999999996, -0.047740000000000005, -0.04712000000000002]，教师结论=различие установлено。
两种降低稀有权重的目标及Q1十倍T均真实重新训练20000updates；未用直接换测试目标替代训练。
最终180项测试通过、467项独立审计通过；需求状态={'PASS': 47, 'FAIL': 0, 'BLOCKED': 0, 'NOT_APPLICABLE': 4}，科学阈值以逐项记录为准。
正式训练7次、累计1924.478秒；正式checkpoint35个，含预算39个；14组SVG/PNG。
全部表格/PNG/SVG及反事实Markdown在独立目录再生成逐字节一致；所有模型/原始样本/轨迹/RNG本地保留并登记SHA256。
详见 [Lab7中文汇总](experiments/lab7/EXPERIMENT_SUMMARY.md)、[复现说明](experiments/lab7/README.md)、[逐项要求](docs/requirements/requirements_lab7.md)、[独立审计](experiments/lab7/final_audit.json)。
Lab0–6原始记录保持不变：Lab4Q2 BLOCKED、Lab5课程split UNVERIFIED、Lab5E2量级FAIL、Lab6E1混合FAIL均保留；前文各阶段段落为历史快照。
