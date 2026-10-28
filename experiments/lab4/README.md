# Lab 4：实验数据与复现入口

本目录保存实验证据和分析产物；按用户要求，不制作正式 LaTeX/PDF 报告。

## 数据与训练协议

沿用 Lab 0 固定 CIFAR-10 索引：train 43000、val 5000、cal 2000、test 10000。自监督阶段只使用 train 图像，cal 保持禁止访问。个人参数见 `profile.json`，原学号仅保存在被 Git 忽略的 `.private/lab4_student_id.json`。

主实验使用 CIFAR ResNet-18，h=512，投影头 512→512→128、无 bias。GPU 协议为 N=256、Efull=100、Eshort=25。每个对比 batch 同时前向 512 个 view，完整计算 510 个负样本；AMP 用于网络，损失使用 float32。没有采用独立小 batch 损失的梯度累积替代。

## 入口

在项目根目录、既有 `.conda-env` 环境下运行：

```powershell
.\.conda-env\python.exe -X utf8 -m reports.tools.lab4_status
.\.conda-env\python.exe -X utf8 -m reports.tools.lab4_run e1
.\.conda-env\python.exe -X utf8 -m reports.tools.lab4_run e2
```

控制器复用已完成的同名正式运行。E3 必须先有真实的 `hypothesis.md` Git 事前提交和 `preregistration_evidence.json`，然后才能执行 `lab4_run e3`。`lab4_run counterfactuals` 执行补充实验。核心训练均调用原有 `ExperimentRunner`。

创建本目录的 `PAUSE` 文件可在完整 epoch 保存 optimizer、scheduler、GradScaler、Python/NumPy/Torch/CUDA/sampler RNG 后停止；用户确认继续后删除该文件并重启对应阶段。恢复要求核心源码和配置一致。已完成的正式运行不得删除或覆盖。

重新生成分析和图表不会重新训练模型：

```powershell
$env:MPLCONFIGDIR='D:/神经网络/tmp/matplotlib-lab4'
.\.conda-env\python.exe -X utf8 -m reports.tools.lab4_results
```

最终审计应在分析和完整测试之后运行 `reports.tools.lab4_audit`；它检查历史文件、checkpoint、原始诊断、标签子集、统计和事前提交。不能仅依据目录存在便认为实验通过。

## 证据布局

- `ledger.json`：编码器训练与初始化，区分 smoke、budget、formal。
- `features.json`、`evaluations.json`：特征提取和分类评价的 RUN_ID、源 checkpoint、选中正则化、指标及计时。
- `E1_sources.json`、`E1.csv`、`E1_probe_checkpoints.csv`：E1 三编码器与五检查点评价。
- `E2.csv`、`E2_diff_runs.json`、`E2_single_selection.json`：六组消融、Lab 0 变量差异核查及仅保留增强的选择依据。
- `hypothesis.md`、`E3_selection.json`、`preregistration_evidence.json`：不可事后修改的 E3 假设与可验证 Git 记录。
- `E3.csv`、`E3_gain.csv`、`E3_statistics.json`：完整 36 格矩阵、三个配对差值和教师规则判定。
- `counterfactuals.json`：反事实问题的实测数值、RUN_ID、解释与阻塞状态。
- `encoders/*/summary.json`、`figures/`：各编码器完整诊断和由原始数据生成的图表。
- `audit.json`、`RUN_registry.json`、`checkpoint_index.json`：最终验收、运行登记和本地 checkpoint 完整性索引。
- `runs/lab4/<RUN_ID>/`：完整配置、环境、原始日志、checkpoint、诊断及缓存。大型 `.pt` 文件保存在本地，按既有规则不加入 Git。

正式结论以实际生成的 `results.json` 和最终 `audit.json` 为准。部分文件只在对应阶段完成后生成，不能把 README 中的产物说明视作完成证明。

## 已知边界

Q2 需要教师《Протокол данных курса》第 4.6 节。项目和已连接资料中未找到该定义，因此保留 BLOCKED，不自造颜色控制数据集。

E2 为单 seed 探索。E3 三个 seed 只改变标签子集与 probe 初始化，不包含编码器预训练随机性；三格 seed=1、r=1 的相同 E1 probe 复用会明确登记。5000 张带标签 val 用于选正则化，使少标签条件较理想化。B1 已用全部 train 标签，不能将其与 SSL 的差异直接解释成“不用标签的代价”；额外低标签监督基线会单独披露数据曝光和更新次数差异。

诊断阈值保持教师原值；有效秩低于 0.2d 不等于分类信息消失。负增益、不成立的统计差异及原始 FAIL 必须如实保留。完成 Lab 4 后停止，不自动进入 Lab 5。

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
