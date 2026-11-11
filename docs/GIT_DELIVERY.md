# Git 交付 / Репозиторий проекта

仓库：<https://github.com/wangzhaofu222-cpu/tpu-neural-networks>

代码、配置、测试、文本运行记录、统计分析与报告通过 Git 保存。原始数据、
Python 环境、全量 checkpoint、MLflow 数据库、缓存和答辩 ZIP 不直接提交。
原始训练提交与事前假设提交保留于现有历史，没有重新初始化或改写历史。

Репозиторий содержит код, конфигурации, тесты, текстовые журналы, результаты
анализа и отчёты. Исходные данные, окружения Python, большие checkpoint-файлы
и локальная база MLflow не включены. История исходных экспериментов сохранена.

## 教师阅读入口

- `src/`：核心算法与统一 Experiment Harness。
- `configs/`：各 Lab 配置；单次实际配置见 `runs/.../config.resolved.yaml`。
- `runs/`：逐次运行记录。训练、预算测试和评价须按 purpose/status 区分。
- `experiments/`：实验汇总、假设、配对统计和审计。
- `reports/labN/LabN_Report.pdf`：当前正式报告。
- `experiments/current_acceptance.json`：后续核查状态入口。
- `docs/MLFLOW_GUIDE.md`：MLflow 展示与重建方式。

仓库初始设为私有。老师需使用获授权的 GitHub 账号访问，或在本机查看。
增加协作者需要老师的 GitHub 用户名；当前不会猜测老师身份或自动发邀请。

Git 历史证明代码与假设的版本顺序；MLflow 用来检索与比较实验。
MLflow 的导入时间不替代原始训练时间和事前提交证据。
