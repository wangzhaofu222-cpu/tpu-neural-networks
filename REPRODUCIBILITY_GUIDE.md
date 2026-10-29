# 可复现性指南（阶段版本）

Lab0正式系列在记录的环境和提交中实测达到A/B/C各自的R3；不保证跨硬件/软件逐位复现。整个课程尚未完成。先查看`EXPERIMENT_SUMMARY.md`、`FINAL_CHECKLIST.md`及`reports/lab0/acceptance.json`。

1. 保留原始10PDF及`docs/requirements/source_inventory.json`。运行`python scripts/verify_audit.py`验证原始文件哈希、页数与条款覆盖。
2. 用`docs/environment/conda-explicit-win64.txt`重建Windows Conda基础环境，再安装`requirements.lock.txt`。锁文件记录完整实际包版本及CUDA wheel源，不等于跨平台保证。
3. 运行`check_environment.py`验证Python/PyTorch/CUDA与硬件。CPU和CUDA测试均实际运行过；不同PyTorch/CUDA/cuDNN/驱动/硬件不能凭同seed宣称逐位复现。
4. 使用profile生成器；仓库仅保存student hash与profile ID，切勿把原学号加入脚本、文件名或公开报告。保留Lab9冲突原字段及项目处理依据。
5. 在项目根运行各CLI，建议统一使用`python -X utf8`处理中文路径；数据路径相对于项目根。数据下载保留官方MD5及archive SHA256；课程协议未取得，split为显式自建版本。当前YAML明确val/cal每类500/200；历史resolved config缺少这两字段时只从已保存split恢复，缺split不猜测，不改变旧config的hash。
6. 运行`python -m pytest -q`。测试临时目录为`tmp/pytest-<UUID>`，避免Windows沙箱临时目录ACL问题；不覆盖旧测试目录。测试使用合成fixture，不能当作CIFAR真实结果。
7. 实验前真实提交代码/配置；Lab1–9在各自主比较前提交hypothesis。尚未写好的主比较假设不能占位提交冒充已预注册。
8. 先运行单epoch预算脚本；完整训练计划单独记录，绝不把预算RUN_ID计入已通过E0。默认2CPU线程/0workers/AMP关；max_seconds用于限时，不隐式缩短并宣称完成。
9. `EXP_ID`按源码Git commit/dirty/source内容hash、resolved config、数据/split、seed计算；环境单列。`RUN_ID`加UUID，每次独立目录。
10. 仅支持epoch边界恢复。总epochs及scheduler计划保持一致，用`--stop-after-epoch`停止，随后新进程`--resume`。完整checkpoint含loader RNG，避免只恢复全局随机数而改变shuffle轨迹。新RUN_ID记录parent并复制截止checkpoint的日志，不覆盖原run。
11. 单独evaluate默认读last.pt和val以核对最终summary；best.pt是不同的选择点，须显式指定并正确解释。test仅用于最终数字，不用于选LR/epoch/主要比较。
12. 小型manifest/config/log/environment/split文本进入Git，checkpoints/plots/datasets不进Git。迁移Colab需要显式复制权重、数据或下载说明、split与校验，不假设远端已存在文件。

Lab0主系列及CF4补充固定在提交`5abedb11663b56690456dca8e8fa3dba24fbb3a7`；完整实验索引为`experiments/lab0/6116b9614f294c4c918500d92471fc4e/ledger.json`。E0独立运行的来源提交另见其manifest。复现历史精确EXP_ID和checkpoint恢复时，应在隔离目录检出该运行记录的提交，再提供原lock、数据和split；不能把后来的发布代码冒充原版本。

主系列Git blob字节重新计算后与原source_hash完全一致，逐文件hash见同目录`source_file_hashes.json`。检出时关闭`core.autocrlf`以保留LF，避免Windows自动行尾转换造成源码字节hash变化。

发布时仅将既定split数量显式放入配置、增加compare表格显示及状态生成工具，并修正测试子进程UTF-8设置。新版本有独立完整测试及真实首epoch兼容检查`docs/environment/release-compatibility.json`；该检查不声称在新提交下重做了全部正式系列。

报告由`reports/tools/summarize_lab0.py`读取原始日志生成，预定E2统计固定为5seed；三个CF4附加seed不回填主CI。注册CPU/GPU性能测量各50预热/100测量，保留全部原始samples及CV标志。八个CF及所有数值均可追溯RUN_ID。

额外GPU长测为独立RUN_ID，N300、B1/B16的CV约20.8%/13.6%，阈值未通过，已写入报告及验收JSON。不得因主N100系列通过而删除或隐去这次诊断，也不宣称短batch延迟具有跨系列稳定性。

`reports/lab0/main.tex`保留完整LaTeX源；本机内置编译进程无法启动（1060），编译未验证。`report.pdf`来自同一俄语文本的ReportLab独立导出并经逐页渲染检查，报告依赖与训练依赖分别锁定。其余Lab的正式实验和报告仍标NOT RUN/BLOCKED。

## Lab1 复现实验

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

## Lab3 数据与分析复现（不制作报告）

统一训练入口仍为ExperimentRunner/src.train，configs/lab3.yaml记录正式参数。原始RUN位于runs/lab3，每条配置/manifest/step日志均已保留。权重只在本地保存，迁移时需连同checkpoints和精确环境复制；checkpoint_index.json记录24个文件的SHA256。不要把Git仓库本身当作已经携带权重。

直接重算Lab3表格、统计与六张图（只写Lab3派生结果，不读取cal、不改Lab2）：

```powershell
$env:MPLCONFIGDIR = "$PWD/tmp/matplotlib-lab3"
.conda-env/python.exe -X utf8 -m reports.tools.lab3_results
.conda-env/python.exe -X utf8 -m reports.tools.lab3_audit
```

训练用100个等长batch构成一个完整dataepoch，按(seed,role,epoch,batch/example)确定性生成；test长度不进入train生成键。整数长度64..128均匀采样，Copy奇数减一。Cal物化后不读取，保留到Lab8。默认严格确定性、float32、explicitattention、无mask/padding。Token损失展平后计算以兼容确定性CUDA。

E1的2000/4000/8000预算扩展仅在固定LR且其余配置完全相同时允许，恢复完整optimizer/scheduler/scaler/RNG，始终新建RUN_ID；正式结果在2000步已满足阈值，因此未扩展。测试验证了边界续训和预算扩展轨迹。历史运行需使用其manifest对应源码；不要在后续源码上强行跳过sourcehash检查。

E3只改变PE；事前hypothesis.md在4d6e559提交，证据见preregistration_evidence.json。seed1和L*使用探索E1，seed2/3在该提交后运行。换seed复现时也改变生成序列；三个配对arm在各自seed中共享完全相同数据。历史hypothesis不回写。

E2和CF5各条件是独立进程，10warmup+30repeat，CUDA同步及resetpeak。E2两实现同float16；CF5两宽度同float32。Flash不可用，实际强制efficientSDPA。五个同进程pilot原始记录保留但不进入主斜率分析。所有量化FAIL与被接受的替代分解分别保存；不能由分析门PASS抹去原始区间FAIL。

<!-- LAB4_AUDITED_STATUS -->
## Lab4 冻结特征实验

使用既有 .conda-env 与 ExperimentRunner，命令和恢复限制见 experiments/lab4/README.md。重算顺序：reports.tools.lab4_results → 完整 pytest（JUnit 路径 docs/environment/tests-lab4-final.xml）→ reports.tools.lab4_audit → reports.tools.lab4_finalize。重算分析不会重新训练模型。E3 假设及 E1/E2 事前快照不可修改；原始 checkpoint 和特征保存在 runs/lab4 并由哈希索引登记。若要重新执行主实验，应使用已登记的源提交及原配置，不把新运行覆盖到旧 RUN_ID。


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
