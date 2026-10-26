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
& .\.conda-env\python.exe -m src.train --config configs/lab0.yaml
& .\.conda-env\python.exe -m src.evaluate --run '<RUN_ID>'
& .\.conda-env\python.exe -m src.compare --runs '<RUN_ID_1>' '<RUN_ID_2>'
& .\.conda-env\python.exe -m src.profile --config configs/lab0.yaml
```

配置覆盖统一用 `--set key=value`，resolved config 保存实际值。Lab0 默认 CPU、deterministic、AMP关、workers0、线程2，默认运行时限120秒，不能误当成完整13次计划。

epoch 边界恢复：

```powershell
& .\.conda-env\python.exe -m src.train --config configs/lab0.yaml --stop-after-epoch 5
& .\.conda-env\python.exe -m src.train --config configs/lab0.yaml --resume 'runs/<RUN_ID>/checkpoints/last.pt'
```

保持原配置的总 epochs，`--stop-after-epoch` 只控制进程停止位置，不能把训练计划 epochs 从5改10，否则调度轨迹不同。恢复创建新RUN_ID并记录父RUN_ID，保存父日志前缀；不会覆盖原结果。当前仅保证**完整epoch边界**恢复，不宣称支持任意batch中断恢复。

## 数据、隐私与 provenance

split在`splits/`，按CIFAR官方train/test命名空间防止“相同数字索引”误判。cal只允许Lab8 conformal读取。train比例仅用于43000train部分。统计量仅从train计算。课程数据协议缺失，不能声称自建split逐索引等同教师文件。

运行目录保存manifest/config/metrics/step-loss/summary/environment；checkpoint含optimizer/scheduler/scaler与Python/NumPy/Torch/CUDA/loader RNG。EXP_ID依据源码、配置、数据/split和seed；RUN_ID附UUID。源码内容hash补足dirty布尔值无法区分两次修改的问题。Git源码状态检查排除生成的运行/划分记录，避免写结果改变下一次实验身份；该范围明确写入manifest。

小文本运行记录进Git；数据、权重、图像、缓存不进Git。Lab9两个冲突profile字段都保留，按用户授权以明确的上游来源字段为准，详情见需求文档。

## 当前范围

已开始公共基础设施与Lab0；Lab1–9只有需求审计和profile规则，**尚未实现/训练/完成报告**。Lab0通过完整验收前不启动后续正式实验。最终提交文件当前为阶段状态记录，绝非已完成课程的声明。
