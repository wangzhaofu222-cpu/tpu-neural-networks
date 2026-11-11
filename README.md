# Нейронные сети — единый экспериментальный проект

## Git и MLflow / Git 与实验展示

Репозиторий: https://github.com/wangzhaofu222-cpu/tpu-neural-networks

双击根目录 `Open_MLflow.cmd` 展示 Lab 0–9 的真实实验记录。
说明与重建：[MLflow 指南](docs/MLFLOW_GUIDE.md)；
教师阅读入口：[Git 交付说明](docs/GIT_DELIVERY.md)。
历史记录导入保留原始 RUN_ID、源码版本与失败状态，不代表重新训练。

Для просмотра результатов Lab 0–9 запустите `Open_MLflow.cmd`.
Исторические записи импортированы с исходными RUN_ID и версиями кода.
Параметры, метрики, кривые обучения и артефакты доступны в MLflow.

2026-11-06 当前验收状态：[current_acceptance.json](experiments/current_acceptance.json)；本轮补充实验与协议：[记录](experiments/revision_20261106/protocol.md)。较早的阶段清单保留为历史快照。

项目按教师 10 份 PDF 建立。当前阶段、已测结果和阻塞见 `EXPERIMENT_SUMMARY.md`；需求矩阵见 `docs/requirements/requirements_matrix.md`。**单元测试、合成 fixture、预算测试均不代表正式实验完成。**

## 环境

使用项目独立 Conda 环境 `.conda-env`，不修改 base/pet/既有 Python3.10 环境。Windows PowerShell 中：

```powershell
& .\.conda-env\python.exe scripts/check_environment.py
& .\.conda-env\python.exe -m pytest -q
```

重建环境：

```powershell
$env:CONDA_PKGS_DIRS = "$PWD\.conda-pkgs"
python -m conda create --prefix .\.conda-env python=3.11 -y --solver classic
& .\.conda-env\python.exe -m pip install -r requirements.lock.txt
```

精确 Python/软件版本以 `docs/environment/current.json`、Conda explicit lock 与 `requirements.lock.txt` 为准；软件参考与数据来源见 `docs/data_sources.md`。

## 命令

```powershell
# 使用真实学号；默认只以profile ID命名输出，绝不保存原号。
& .\.conda-env\python.exe -m src.profile_gen --student-id '<ID>' --out configs/profiles/current.json
& .\.conda-env\python.exe scripts/download_cifar10.py
# 只测一个完整CPU epoch，不自动启动长训练。
& .\.conda-env\python.exe scripts/budget_lab0.py
# 预算门通过后手动安排完整实验；purpose=formal要求源码/配置已提交且干净。
& .\.conda-env\python.exe -X utf8 -m src.train --config configs/lab0.yaml --set experiment.purpose=formal --set runtime.max_seconds=600
& .\.conda-env\python.exe -m src.evaluate --run '<RUN_ID>'
& .\.conda-env\python.exe -X utf8 -m src.compare --runs '<RUN_ID_1>' '<RUN_ID_2>' --format markdown
& .\.conda-env\python.exe -m src.profile --config configs/lab0.yaml
```

配置覆盖统一用 `--set key=value`，resolved config 保存实际值。Lab0 默认 CPU、deterministic、AMP关、workers0、线程2，默认运行时限120秒，不能误当成完整13次计划。

epoch 边界恢复：

```powershell
& .\.conda-env\python.exe -X utf8 -m src.train --config configs/lab0.yaml --set experiment.purpose=formal --set runtime.max_seconds=600 --stop-after-epoch 6
& .\.conda-env\python.exe -X utf8 -m src.train --config 'runs/<RUN_ID>/config.resolved.yaml' --resume 'runs/<RUN_ID>/checkpoints/last.pt'
```

保持原配置的总epochs=12，`--stop-after-epoch`只控制进程停止位置，不能把训练计划epochs从6改12，否则调度轨迹不同。恢复创建新RUN_ID并记录父RUN_ID，保存父日志前缀；不会覆盖原结果。仅保证**完整epoch边界**恢复。历史checkpoint须使用其manifest记录的源码；源码改变时明确拒绝继续，不能修改旧manifest规避检查。

## 数据、隐私与 provenance

split在`splits/`，按CIFAR官方train/test命名空间防止“相同数字索引”误判。cal只允许Lab8 conformal读取。train比例仅用于43000train部分。统计量仅从train计算。课程数据协议缺失，不能声称自建split逐索引等同教师文件。

运行目录保存manifest/config/metrics/step-loss/summary/environment；checkpoint含optimizer/scheduler/scaler与Python/NumPy/Torch/CUDA/loader RNG。EXP_ID依据源码、配置、数据/split和seed；RUN_ID附UUID。源码内容hash补足dirty布尔值无法区分两次修改的问题。Git源码状态检查排除生成的运行/划分记录，避免写结果改变下一次实验身份；该范围明确写入manifest。

小文本运行记录进Git；数据、权重、图像、缓存不进Git。Lab9两个冲突profile字段都保留，按用户授权以明确的上游来源字段为准，详情见需求文档。

## 当前范围

Lab0已有完整真实训练、反事实测量与六节俄语报告，原有证据保持不变；其历史LaTeX编译未验证记录仍保留。

Lab1已复用同一个ExperimentRunner完成实现、E1/E2/E3、5个反事实问题与可训练性地图。最终47项测试通过，俄语报告9页，实际LaTeX编译和逐页视觉检查通过。**数值阈值未全部通过**：缩放残差L96的信号比例6.9964超出上限6.8493约2.15%，保留FAIL并用逐块方差与交叉项实测解释。α=1的4个训练条件发生真实数值失败，日志完整保留，不冒充三轮成功。报告和逐项状态见`reports/lab1/report.pdf`、`main.tex`、`acceptance.json`。

Lab2已完成标准模型、正式实验、多seed统计、反事实及checkpoint保存，新报告按用户要求暂缓，证据见`experiments/lab2/README.md`。Lab3前独立核查通过：70测试、38checkpoint及原始统计重算一致，原记录不变，见`experiments/lab2_verification/audit.json`。

Lab3已完成Copy/L0=128/深度2的全部必做实验、三seed配对、注意力诊断和五题反事实；88项测试通过。E2原始显存斜率区间FAIL保留，同时完成PDF允许的替代成本分解；详情见`experiments/lab3/README.md`、`results.json`和`docs/requirements/requirements_lab3.md`。本轮没有生成或修改正式报告。Lab4–9未启动；课程协议缺失导致的具体阻塞仍逐项保留。复现和Git事前假设要求见`REPRODUCIBILITY_GUIDE.md`，完整状态以`FINAL_CHECKLIST.md`为准。

<!-- LAB4_AUDITED_STATUS -->
## Lab4

实验已推进至 Lab4，Q2 因缺教师 Protocol §4.6 保留 BLOCKED；不虚报全部完成。复现入口与数据索引见 [Lab4 README](experiments/lab4/README.md)，逐项验收见 [Lab4 audit](experiments/lab4/audit.json)。没有制作新报告，没有启动 Lab5。


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
