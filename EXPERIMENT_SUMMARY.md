# 实验项目阶段汇总

这是Phase A/B与Lab0开发阶段记录，不是课程完成声明。

## 已完成的实证工作

- 10 PDF/152页全文审计、2153个原文条款ID；原PDF哈希不变校验PASS。
- 五份需求文档、逐Lab原文登记、可追踪主矩阵、数据依赖/缺失/歧义登记。
- 独立Conda Python3.11.17 / torch2.7.1+cu126 / torchvision0.22.1+cu126；依赖检查通过；CPU/CUDA真实运算检查通过。
- Profile `P3adf02`：train_fraction=0.25、width=96、depth=4、batch=64、adamw、cosine。未保存原学号。
- 公共runner/config/manifest/完整epoch边界checkpoint/JSONL/compare/独立evaluate/profiling已实现。
- 最新测试：21/21通过，failures=0、errors=0、skipped=0，18.786秒；证据`docs/environment/tests-lab0.xml`。
- 测试真实验证新进程resume参数相对差为0、逐step loss一致，CPU/CUDA均运行；全部训练测试是明确标记的fixture，不能算正式实验。
- CPU与CUDA前向性能数据见`docs/environment/profile-*-development.json`，各B≥10预热/30测量；有独立RUN_ID。它们是开发诊断，不是完整正式E4验收。

## 真实训练预算

真实 CPU 单epoch 23.93 秒；13×10epoch估计 0.86小时，不含CUDA附加实验与诊断。教师60秒epoch目标：达到。预算RUN_ID `97838186ca30f4d648d8-3e8078d5292648fa9226c10255b682ed`，不是已完成E0。

## 课程验收状态

| Lab | 实现 | 正式实验 | 俄语报告 | 当前限制 |
|---|---|---|---|---|
| 0 | 公共框架与基础测试PASS；完整实验编排/CF/自动报告尚未完成 | NOT RUN | NOT RUN | 预算与数据协议；未完成至少13次完整训练 |
| 1–3 | NOT RUN；profile/需求已登记 | NOT RUN | NOT RUN | 先通过Lab0 |
| 4 | NOT RUN；profile/需求已登记 | BLOCKED（完整验收CF2） | NOT RUN | 缺课程协议§4.6颜色控制集；主实验仍需上游完成 |
| 5 | NOT RUN | BLOCKED（正式mask核验） | NOT RUN | 缺课程Planetoid masks |
| 6–7 | NOT RUN；profile/需求已登记 | NOT RUN | NOT RUN | 先通过Lab0，长链/采样/CF需另预算 |
| 8 | NOT RUN | NOT RUN | NOT RUN | 尚无Lab2真实checkpoint；nearOOD附加缺11类名单 |
| 9 | NOT RUN | BLOCKED | NOT RUN | 缺RAVDESS课程预处理包/演员划分；encoder冲突已按用户授权处理 |

## 仍需明确的事实

课程协议不存在；不将自建split声称为教师标准逐索引split。CIFAR官方页面未列独立许可证，已记录，不声称完成全部许可审查。CPU温度未测，用户已要求忽略等待项；预算/线程限制仍保留。没有启动无人值守通宵任务。

后续：真实CPU单epoch预算门→补齐Lab0正式编排/CF与报告工具→按可行预算正式实验并验收；Lab0通过后逐个Lab推进。受缺失资源影响的部分等待教师材料，其余不虚报完成。
