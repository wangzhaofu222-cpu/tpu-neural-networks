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
