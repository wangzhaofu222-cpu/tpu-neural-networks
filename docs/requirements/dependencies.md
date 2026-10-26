# 依赖与数据权限

1. Lab 0 的 config、seed、split、manifest、runner、checkpoint、日志、compare、profiling 被所有后续实验复用；后续任务扩展模型/数据/任务适配器，不建立第二套训练引擎。
2. CIFAR-10 官方 train 50000：每类 val 500、cal 200，剩余 train 4300；官方 test 10000 单独命名空间。Lab 0 创建一个共享 split manifest，Lab 2/4/8 直接引用其 hash。训练 seed 变化不重新划分 split。
3. `train_fraction` 仅作用于 train 43000，不改变 val/cal/test；Lab 2 使用自己 profile 的嵌套分层 fractions，不沿用 Lab 0 的训练比例。
4. train 用于学习与统计归一化；val 用于选超参数/模型/温度；test 仅最终评价。cal 在 Lab 8 的附加 conformal 任务之前禁止读取；温度用 val，绝不能用 cal。Lab 9 仍拒绝 cal。
5. Lab 1 的非侵入式监控、rho_T/GI 在 Lab 5 复用。Lab 2 的 r=1 两模型 checkpoint 用于 Lab 8；Lab 4 的 encoder + probe 为允许备选。Lab 2/4 编码器与 Lab 8 扰动用于 Lab 9。
6. Lab 6 合成分布/生成指标可供 Lab 7 扩展；注意 Lab 6 不标准化、Lab 7 必须按 train 标准化且评价回原单位。Lab 7 与 Lab 6 三模型直接比较是附加项，需相同目标/尺度。
7. Lab 5 读课程 Planetoid masks 并与库标准 masks 对照；全图特征/边可见，test 标签不进入训练/选择。cal 为空。
8. Lab 9 必须有课程 RAVDESS clip manifest、actor split、face_found、音视频对应/偏移、编码器 checkpoint hash。不能凭 14/3/2/5 人数自行猜演员名单。encoder字段冲突已按用户授权以visual_encoder来源字段优先，保留冲突原始值及处理依据。
9. 每个下游运行记录 source RUN_ID、checkpoint 内容 hash、split hash、feature cache hash；迁移 Colab 需显式打包这些依赖，不能假设远端已有本地文件。

教师 Lab 0 的“cal 在任何工作到 Lab 8 前均不读取”与 Lab 3 需创建但保留 cal 一致；Lab 6/7 不创建 cal，不补造。
