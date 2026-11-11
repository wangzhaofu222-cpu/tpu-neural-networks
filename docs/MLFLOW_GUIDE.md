# MLflow 展示指南 / Руководство по MLflow

## 打开与展示

在项目根目录双击 `Open_MLflow.cmd`，浏览器打开 <http://127.0.0.1:5000>。
服务只监听本机。关闭服务窗口会停止展示服务，数据库和实验文件仍然保留。

Сервер запускается файлом `Open_MLflow.cmd`. Интерфейс доступен только на этом
компьютере: <http://127.0.0.1:5000>. Остановка сервера не удаляет результаты.

1. 在 Experiments 选择 Lab 0–9。每个 Lab 是一个独立 experiment。
2. 打开 `00 - Lab N results, figures and report` 查看汇总、图表、报告及原始分析文件。
3. 其他条目对应原始 RUN_ID 的真实训练、评价或诊断。
4. 在运行列表添加参数列，如 `model.name`、`experiment.seed`、`optimizer.lr`。
5. 选中需要比较的运行，再使用 Compare。训练曲线以 `step.train_loss`、
   `epoch.train_loss`、`epoch.validation_loss` 等命名。
6. 最终指标以 `result.` 开头，保留原文件字段层次，例如
   `result.validation_metric`、`result.metrics.accuracy`。不同任务不可混为同一指标。
7. Artifacts / original 是未修改的配置、日志、manifest 和 summary。
   provenance 保存原文件哈希及原模型的路径与哈希。

常用筛选 / Полезные фильтры：

```text
tags.`report.referenced` = 'true'
tags.`source.purpose` = 'formal'
tags.`source.kind` = 'training'
tags.`source.status` = 'COMPLETED'
```

报告引用筛选可能排除模型训练来源：部分报告引用的是后续评价运行；查看
`source.source_RUN_ID` / `source.encoder_RUN_ID` 等标签追溯其训练来源。

## 导入的含义与保真规则

- 这是对已完成实验的历史记录导入，不声称过去运行时使用了 MLflow。
- `source.RUN_ID` 是原项目运行编号；MLflow 自己还会生成一个不同的 run ID。
- `source.git_commit` 是原实验源码版本，不替换为本次导入版本。
- 原始开始时间保留；结束时间只有在原记录提供时间或耗时时才恢复。
  缺少每步时间时不会虚构时间轴，曲线应按 Step 查看。
- 每条日志流最多展示 1000 条等距选出的真实记录，包含首尾；没有插值、
  平均或更改原值。完整 JSONL 原文件保存在 Artifacts。
- 未完成和失败运行保留原状态。导入完成不表示教师验收通过；验收文件见
  各 Lab 汇总条目的 analysis 和 `experiments/current_acceptance.json`。
- 模型 checkpoint 路径及 SHA256 保存于 `provenance/checkpoints.json`；
  全量模型仍位于本机 `runs`，没有把全部大模型复制进 Git。
- 重新运行 `Sync_MLflow.cmd` 可同步新记录。内容哈希相同的运行不会重复导入；
  历史文件变化会产生新导入版本，旧版本保留。

Это импорт исторических экспериментальных записей, а не повторное обучение.
Оригинальные RUN_ID, версии Git, статусы, параметры и значения метрик сохранены.
Для графиков выбираются реальные точки без интерполяции; полные журналы доступны
в Artifacts. Наличие записи в MLflow не означает прохождение всех критериев курса.

## 新训练与 MLflow

原训练环境与展示环境独立。新训练可使用以下入口：

```powershell
& .\.conda-env\python.exe scripts/train_with_mlflow.py --config configs/lab0.yaml
```

它调用原有 `src.train` 和 ExperimentRunner，运行结束后使用单独解释器同步记录。
训练失败也会尝试导入已生成记录；同步失败不会删除训练输出。
恢复训练参数与原入口相同。现有正式训练预算和配置不会被此入口自动修改。

## 在另一台 Windows 电脑重建

先依照项目原文档准备训练所需数据和环境。仅展示已有实验时，MLflow 无需
PyTorch，可单独安装：

```powershell
python -m venv .mlflow-env
& .\.mlflow-env\Scripts\python.exe -m pip install -r requirements-mlflow.lock.txt
& .\.mlflow-env\Scripts\python.exe scripts/mlflow_sync.py
& .\.mlflow-env\Scripts\python.exe scripts/mlflow_ui.py
```

`mlflow_store` 包含 SQLite 和 artifact 文件，不提交到 Git。已有本机库无需重新导入。
从 Git 克隆后可由保存在仓库中的文本运行记录重新构建展示；大模型和原始数据需另行准备。
SQLite 中保存本机 artifact 绝对路径，迁移电脑推荐重新导入，不直接复制数据库。

## 官方说明

- [MLflow Tracking](https://mlflow.org/docs/latest/tracking/)
- [Tracking Server](https://mlflow.org/docs/latest/tracking/server/)
