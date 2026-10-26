# 可复现性指南（阶段版本）

当前不能宣称整个课程已完成或已达到正式R0–R3。先查看`EXPERIMENT_SUMMARY.md`与`FINAL_CHECKLIST.md`中的实际状态。

1. 保留原始10PDF及`docs/requirements/source_inventory.json`。运行`python scripts/verify_audit.py`验证原始文件哈希、页数与条款覆盖。
2. 用`docs/environment/conda-explicit-win64.txt`重建Windows Conda基础环境，再安装`requirements.lock.txt`。锁文件记录完整实际包版本及CUDA wheel源，不等于跨平台保证。
3. 运行`check_environment.py`验证Python/PyTorch/CUDA与硬件。CPU和CUDA测试均实际运行过；不同PyTorch/CUDA/cuDNN/驱动/硬件不能凭同seed宣称逐位复现。
4. 使用profile生成器；仓库仅保存student hash与profile ID，切勿把原学号加入脚本、文件名或公开报告。保留Lab9冲突原字段及项目处理依据。
5. 在当前工作目录运行各CLI；数据路径相对于项目根。数据下载保留官方MD5及archive SHA256；课程协议未取得，当前split为显式自建版本，需在报告注明。
6. 运行`python -m pytest -q`。测试临时目录为`tmp/pytest-<UUID>`，避免Windows沙箱临时目录ACL问题；不覆盖旧测试目录。测试使用合成fixture，不能当作CIFAR真实结果。
7. 实验前真实提交代码/配置；Lab1–9在各自主比较前提交hypothesis。尚未写好的主比较假设不能占位提交冒充已预注册。
8. 先运行单epoch预算脚本；完整训练计划单独记录，绝不把预算RUN_ID计入已通过E0。默认2CPU线程/0workers/AMP关；max_seconds用于限时，不隐式缩短并宣称完成。
9. `EXP_ID`按源码Git commit/dirty/source内容hash、resolved config、数据/split、seed计算；环境单列。`RUN_ID`加UUID，每次独立目录。
10. 仅支持epoch边界恢复。总epochs及scheduler计划保持一致，用`--stop-after-epoch`停止，随后新进程`--resume`。完整checkpoint含loader RNG，避免只恢复全局随机数而改变shuffle轨迹。新RUN_ID记录parent并复制截止checkpoint的日志，不覆盖原run。
11. 单独evaluate默认读last.pt和val以核对最终summary；best.pt是不同的选择点，须显式指定并正确解释。test仅用于最终数字，不用于选LR/epoch/主要比较。
12. 小型manifest/config/log/environment/split文本进入Git，checkpoints/plots/datasets不进Git。迁移Colab需要显式复制权重、数据或下载说明、split与校验，不假设远端已存在文件。

对于当前尚未执行的正式实验、反事实测量及报告，状态一律NOT RUN/BLOCKED；不存在可引用的正式RUN_ID。
