# 实验项目阶段汇总

Phase A/B已完成；Lab0阶段验收：**PASS**。整个Lab0–Lab9课程项目尚未完成，不能作为全课程完成提交。

## 已完成

- 10份教师PDF、152页全文审计、2153个原文条款ID；原PDF哈希不变。2153包含理论和参考文献，不等于2153项必做实验。
- 项目独立Python3.11.17 / PyTorch2.7.1+cu126 / torchvision0.22.1+cu126，精确锁定依赖；CPU与RTX3060 Laptop CUDA实测可用。
- 公共runner、五CLI、哈希profile、数据角色隔离、完整checkpoint/RNG/GradScaler、逐step日志、独立evaluate、比较与性能测量。
- 自动测试25/25通过，errors=0、failures=0、skipped=0，20.400秒。证据：`docs/environment/tests-lab0-release.xml`。合成fixture测试仅验证实现。

## Lab0真实结果

- 真实CPU预算24.09秒/epoch，12epoch；13个逻辑训练条件估计62.63分钟。主serial-suite实际65.05分钟；此前E0/失败尝试、后续CF4补充与注册profiling另计。
- E0 loss 1.964259→0.895176，val accuracy=0.6232，独立evaluate完全吻合。首次K10的loss比值0.500332未通过，保留FAIL记录；K12从头重训通过。
- A/CPU、B/CUDA、C/CUDA+AMP：E1逐step loss与最终指标重复一致；E3完整进程重启恢复后参数相对差均为0。有效optimizer steps A/B为2016，C为2015；C实际有1次overflow跳步。
- E2预定5seed：mean=0.633440，样本std=0.008708，95% Student t CI=[0.622628,0.644252]；2std参考效应=0.017416。
- workers2配对与workers0基准的全部2016个step losses完全相同。只恢复权重的最终参数相对差=0.371028，不能当成正确恢复。
- CF4额外3个seed单独登记，不重算预定5seed统计；数据1字节干预和optimizer/lr/RNG诊断均有真实记录。
- E4主CPU/GPU系列各50预热/100测量，全部B的CV≤10%；附加GPU N300诊断在B1/B16得到CV约20.8%/13.6%，阈值状态FAIL，已保留并在报告披露，不能把主系列通过扩展为普遍延迟稳定保证。
- 俄语六节报告：`reports/lab0/report.pdf`、`main.tex`、`report.ru.md`；图表、RUN_ID注册表、实验验收及构建状态同目录。
- 主E1/E2/E3与补充训练固定提交`5abedb11663b56690456dca8e8fa3dba24fbb3a7`。发布时显式split参数及比较显示的改动另有兼容验证；不改写历史manifest。

## 课程状态

| Lab | 实现/正式实验/报告 | 限制 |
|---|---|---|
| 0 | PASS | PDF经渲染检查；LaTeX编译器启动1060，tex编译未验证 |
| 1 | 完成：实现、正式E1–E3、5CF、已编译俄语报告 | 缩放残差L96信号上界FAIL，已实测诊断；α=1数值失败保留 |
| 2 | 正式实验完成；新报告按用户要求暂缓 | 逐项状态见 experiments/lab2/results.json |
| 3 | 全部必做实验完成；报告按用户要求暂缓 | E2原始显存斜率区间FAIL；已完成PDF允许的替代成本分解 |
| 4 | NOT RUN；CF2完整验收BLOCKED | 缺课程协议§4.6颜色控制集 |
| 5 | NOT RUN；正式masks核验BLOCKED | 缺课程Planetoid masks |
| 6–7 | NOT RUN；需求/profile已登记 | 长链、训练、采样与CF须各自先做真实预算 |
| 8 | NOT RUN | Lab2 checkpoint已保存；近OOD附加缺11类名单 |
| 9 | BLOCKED（正式数据） | 缺人脸帧/音频预处理包、演员/clip划分；编码器已按用户授权选cnn_lab2 |

## 已知边界

课程数据协议缺失，自建split不声称逐索引等同教师文件。CIFAR官方页未列独立许可证，不分发原数据。CPU温度未测，用户要求不等待传感器。断点恢复保证仅限完整epoch边界和记录的代码/环境。没有遗留后台或通宵训练计划。LaTeX编译未验证与PDF渲染验收分开记录。

## Lab1 阶段：工作完成，存在已解释的数值阈值失败

- 复用 Lab0 ExperimentRunner；新增深层MLP、精确残差块和激活/梯度监测。原Lab0的374个报告、记录与检查点文件逐字节保留。
- Profile：深度8/24/48/96、Gaussian σ=.02、SGD momentum；CIFAR-10全train43000、宽256、3epochs，固定网格选择LR=0.003。
- E1完整12组初始化及4组训练一轮后复测；最大深度全部前/反χ误差≤20%。E2完整12条件；α=1四组真实数值失败，保留失败记录；其余条件完成3整轮。缩放残差L96的q比例6.9964超过上限6.8493约2.15%，该数值阈值为FAIL；逐块c和交叉项实测已定位近似条件失效，其余增长阈值通过。
- E3假设先真实Git提交，再运行共同seeds1/2/3。rho(A)-rho(B)=[0.2834983102697599, 0.2876807579434426, 0.3025880211525119]；均值0.291256、样本std0.010034；差异判定：成立。只适用于同一数据划分。
- 自动测试47项全通过；报告9页，真实LaTeX编译与每页图像检查均通过。运行累计28.24分钟（含准备、小测试和预算；开发、测试和报告编译另计）。
- 证据：reports/lab1/acceptance.json、results.json、report.pdf、main.tex、experiments/lab1/ledger.json、hypothesis.md。


## Lab2 阶段（报告按用户要求暂缓）

- 复用公共 ExperimentRunner，完成标准 ResNet18/ViT、E1 五比例、E2 三扰动、E3 三配对 seed、地图及全部五个反事实问题。
- 完成 19 个正式训练运行，逻辑轨迹共 180506 次优化器更新；已记录训练耗时下界 7.398 小时。复用记录不重复计数。
- 曾有1个外部中断父进程，从完整checkpoint精确恢复；499个已记录更新被重放。20次正式进程调用中19次完成，记录的实际更新含重放为181005次。中断父进程精确总时长不可恢复，不能伪报精确累计时长。
- 自动测试 70 项通过；E3 平均差 -0.249467，样本标准差 0.017253，差异判定：成立。
- 完整 PASS/FAIL/边界：experiments/lab2/results.json；原始 ledger、独立评估 RUN_ID、图表、反事实和 checkpoint 索引同目录。
- 已保留全量模型，解除 Lab8 对 Lab2 checkpoint 的依赖；Lab8/9 尚未启动，其他数据缺口仍保留。未生成新报告，未启动 Lab3。

## Lab3 阶段（历史记录，报告暂缓）

- 开始前独立核查Lab2：70测试通过、38checkpoint可加载、主要表格/统计逐字节重算一致；1538个历史文件保持不变。结果存于experiments/lab2_verification，原Lab2记录未重写。
- 个人任务Copy，L0=128，N=2，d128/h4/FFN256/B32，AdamW固定LR=.003，四种PE均2000steps；LR仅按sinusoidal的valL0选择。
- E1两项验证阈值PASS：none=.057861328125，bestPE−none=.40338134765625。L128/256/512/1024及额外P1评估完整；外推明显退化，未据定义域推断准确率保证。
- E2显式/forced-efficient两后端各五长度，独立进程真实测量。显存全网格α=.435671/.127139（以results.json精确值为准），最长两点α=1.086481/.259660；原始区间FAIL保留。PDF§9.2允许的二次+线性+常数分解已完成，不能声称实测进入纯渐近区间。Flash不可用证据单列，未冒充支持。
- E3假设事前提交4d6e559；L*=256，RoPE−absoluteP2三差值[-.0882568359375,-.07147216796875,-.056610107421875]，均值−.072113037109375，样本SD=.01583309481361658。三者同负且absmean>=2SD，差异成立，负向假设获得支持。seed1来自探索E1，其选择局限已披露。
- 正式训练10次（选LR3+E1新增3+E3新增4），累计293.600067秒；含成功smoke/budget为12次、295.345164秒，失败smoke另保留。24个checkpoint（正式20），88项测试通过；6张科学图经视觉检查，五个反事实问题及按block/head注意力统计均已保存。
- 当前证据：experiments/lab3/README.md、results.json、audit.json、checkpoint_index.json。该阶段没有生成或修改LaTeX/PDF报告；该阶段结束时尚未启动Lab4。

<!-- LAB4_AUDITED_STATUS -->
## Lab4 最终证据状态：BLOCKED

- Lab3 独立复核 PASS：88 项测试重跑，1908 个历史文件不变，原有 E2 两项 FAIL 保留。
- Lab4 E1/E2/E3 均已执行，完整 36 格 E3 矩阵包括 3 格明确复用的 E1 结果。E3 事前 Git：`57f3fd3d7fe2d8f39c5f28efb73f2b2f9e363e41`。
- E1 probe：B0 43.57%、B1 89.88%、SimCLR 76.62%；G(1)=+33.05 个百分点。
- E3 rmin 三差值：[0.3741, 0.377, 0.3677]，均值 0.372933333，样本 SD 0.004758501，教师判定：различие установлено。事前点预测 0.165250，误差 +0.207683333；方向成立不等于数值预测准确。
- 106 项自动测试全部通过。四个反事实问题有实测依据；Q2 缺教师 Protocol §4.6，保留 BLOCKED。因此不能宣称全部必做任务已无条件通过。
- 正式编码器训练 10 次，probe 正则化候选训练 144 次；B0 仅初始化。编码器累计训练 2.5510 小时；完整非重叠计时见 audit.json。
- 本地 checkpoint 共 334 个，其中正式 325 个；所有配置、原始记录、特征缓存、哈希及路径均已登记。大型权重和缓存不强制加入 Git。
- 按用户要求不制作或修改正式 LaTeX/PDF 报告；Lab5 未启动。全部详细数值与 PASS/FAIL/BLOCKED 见 experiments/lab4/results.json、audit.json、counterfactuals.json。


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
