# Нейронные сети — единый экспериментальный проект

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
