# 初次 E0 未通过与修正计划

真实RUN_ID：`187f3bd4a93cd598f53b-47b8eee4daa145949b8406914abff222`，Git `48d4d3f`，CPU确定性模式A，10epoch，profile P3adf02。未使用test调参。

first train loss=1.9642590621016747，final=0.982782258809999，ratio≈0.500332；要求final≤0.9821295310508374，因此loss条件严格FAIL，不四舍五入成PASS。最终validation accuracy=.601，达到≥.30条件。所有10epoch和1680optimizer steps完整记录。

loss轨迹持续下降，第9epoch=1.0071873082338378、第10epoch=.982782258809999；末epoch学习率3.0153689607045842e-5。没有NaN/Inf、数据形状错误或随机基线失败证据；测试已验证梯度与checkpoint流程。待检验解释：在cosine衰减下优化步数略不足，不是据test标签做模型选择。

修正：保持profile的fraction/width/depth/batch/optimizer/scheduler与LR/weight decay/数据split/seed不变，把总训练计划K从10改为12，完整从头重训，不沿用旧scheduler假装同一轨迹。Lab0未固定K必须为10，§12.2要求按实测成本选择K；10是预算示例。约13×12×23.93s=3732s≈62.2min，CUDA与诊断另计。新结果须独立重新验收，尚不保证通过。

旧run、checkpoint、原始日志全部保留；该FAIL不计入后续PASS结果，也不删除。
