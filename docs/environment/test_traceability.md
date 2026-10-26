# Lab 0 自动化测试追踪

最新完整证据：`tests-lab0.xml`，21 tests，0 failures/0 errors/0 skipped，18.045秒；实际包含CPU与CUDA运行。原先临时目录ACL/父目录问题、CUDA设备字符串断言失败的XML另存，未删除。最终测试检查的是设备类型，`cuda:0`合法且不被误判为CPU。

| 教师§25编号 | 实际测试 | 覆盖 |
|---|---|---|
| L0-T01 | test_L0_T01_config_validation | resolved字段、加载一致、缺失section拒绝 |
| L0-T02 | test_L0_T02_all_roles_repeated_and_leak_free | 四角色、固定seed、43k/5k/2k/10k与命名空间 |
| L0-T03 | test_L0_T03_checkpoint_output_unchanged | save/load后output逐位一致 |
| L0-T04 | test_L0_T04_metrics_deterministic | 相同输入评价完全一致 |
| L0-T05 | test_L0_T05_T06_T08_ids_and_code_changes | 改lr、data/source改变EXP_ID |
| L0-T06 | test_L0_T06_T07_repeat_no_overwrite | 同条件EXP相同/RUN不同、逐step loss相同 |
| L0-T07 | test_L0_T06_T07_repeat_no_overwrite | 原目录小文件byte内容未被第二次运行修改 |
| L0-T08 | test_L0_T05_T06_T08_ids_and_code_changes | 实际源码文件变化被source hash捕获 |
| L0-T09 | test_L0_T09_profile_rules_and_unicode | SHA256逐字段独立公式校验、Lab1–9选项、大小写/Unicode |
| L0-T10 | test_L0_T10_two_profiles_without_code_edit | 两个测试标识各自产生配置和真实训练，无源代码修改 |
| L0-T11 | test_L0_T11_cpu / test_L0_T11_cuda_selected_by_config | 配置选择CPU/CUDA、GPU真实完成训练step |

额外覆盖：新进程epoch边界resume逐step loss和最终参数完全相同；Python/NumPy/Torch/loader RNG恢复；独立evaluate CLI；嵌套均衡子集；cal拒绝；数据改1byte hash变化；真实profiling测量且不改参数；Student t CI与三对差值判断；合成fixture无法当作正式课程实验验收。

全部训练测试使用清楚标记的合成fixture，只证明实现，不满足CIFAR正式实验。测试没有验证完整E0–E4结果、13次正式训练或53个反事实答案。Lab1–9特定测试尚未实现。
