# 课程需求审计

审计日期：2026-10-26（Asia/Novosibirsk）。原始目录只有 10 份 PDF，共 152 页；没有已有代码、数据、配置、checkpoint、Git 历史或数据协议。原始 PDF 保留不动，SHA256、页数与大小见 `source_inventory.json`。全文提取见 `sources/`，逐段登记见 `lab0_clauses.md` 至 `lab9_clauses.md`。已通读所有提取文本，并渲染核对每份 PDF 的关键公式/表格页；不是对 152 页逐页图像审校。

Lab 0 为 2026-10-09 修订 2.1；Lab 1–9 为 2026-10-08 修订 4。实施标准是原始 PDF。用户概述与 PDF 不一致时使用 PDF。

| Lab | 核心交付 | 正式主要比较 | 前置 Git 假设 | 报告章节数 |
|---|---|---|---|---:|
| 0 | 统一 harness、provenance、E0–E4、R0–R3 | 同 seed / 多 seed / 真正进程重启 resume | 未明确要求 hypothesis 提交 | 6 |
| 1 | 信号/梯度、初始化、残差、trainability map | E3：两配置，seeds 1,2,3 | E3 前 | 6 |
| 2 | CIFAR ResNet-18 vs ViT、固定步数、shift | E3：两架构，seeds 1,2,3 | E3 前 | 6 |
| 3 | 四位置方案、外推、复杂度、attention | E3：两位置方案，长于训练长度 | E3 前 | 6 |
| 4 | 自写 NT-Xent、SimCLR、probe/kNN、消融 | E3：标注子集/probe seeds；不是重训编码器 | E3 前 | 6 |
| 5 | MLP/GCN/第二 GNN、null graph、oversmoothing | **E2**：原图 vs 保持度数 null，三 seeds | **E2 前** | 6 |
| 6 | 受约束 EBM、Langevin、affine flow | E3：参数匹配两模型、三 seeds | E3 前 | **5** |
| 7 | 自写 schedule/DDPM/DDIM、exact score | E3：Dmode，三 seeds | E3 前 | 6 |
| 8 | 既有 checkpoint、温度、攻击、OOD、拒答 | **E2**：温度在 clean/shift 的收益差 | **E2 前** | 6 |
| 9 | actor split、缓存编码器、fusion、干预 | **E1**：融合 vs 参数匹配单模态控制 | **E1 主比较前** | 6 |

Lab 1–9 主要比较通用判断：三对差值全部同号且 `abs(mean(delta)) >= 2*sample_std(delta)`，否则写“差异未建立”；保留三个原始差值。Lab 6 另要求超过指标噪声。所有图表由真实日志产生，每个数值追溯 RUN_ID。负结果可接受，但需定位并验证原因。

必答反事实：Lab 0 有 8 题，其余每 Lab 5 题，共 53 题；每题必须引用自己的测量/RUN_ID。`Граница применимости` 中的额外运行也是正文要求，不能当成附加题跳过。`Дополнительные задания` 单独登记，不自动列为必做。

已确认学号由用户私下提供。仅写入哈希与 profile ID；不采用 PDF 的示例学号、示例 profile 数值。不把本对话或原始学号复制进仓库。

尚未达到任何 Lab 完成条件。需求审计与基础测试通过不代表 E0–E4 或后续 E1–E3 完成。
