# 实施顺序、预算及验收门

## 当前阶段

Phase A：全文与关键图像审计、逐条原文登记、依赖和阻塞记录。Phase B：创建项目独立 Python≥3.11、PyTorch 2.x 环境，检查 CUDA、锁定实际完整依赖；基础 harness 开发。未完成 Phase C 正式实验，不能进入 Lab 1 正式训练。

## 顺序

1. A-01 盘点 10 PDF/hash/页数；A-02 逐章登记；A-03 profile/冲突；A-04 数据依赖；A-05 预算计划。
2. B-01 隔离 Conda 环境；B-02 精确 lock + 环境诊断；B-03 Git 初始化（不伪造提交身份）；B-04 公共 CLI/runner/provenance/split/checkpoint/profiling；B-05 11 项 Lab 0 必测 + 轨迹恢复/数据泄漏集成测试。
3. C-01 合成 fixture 仅作测试；C-02 真实 CIFAR 一个 epoch 计时/资源测试；C-03 提交正式预算；C-04 E0 完整 sanity；C-05 按计划 E1/E2/E3、CF 干预、E4；C-06 自动阈值验收；C-07 6 节俄语报告/8 CF 答案/PDF 编译检查。
4. D-01..D-09 严格逐 Lab 实现/测试/短测试/预算/正式实验/验收/报告，Lab 5 与 8 在 E2 前、Lab 9 在 E1 主比较前提交 hypothesis；其余 E3 前。
5. E-01 对所有条款证据审计，输出 FINAL_CHECKLIST、EXPERIMENT_SUMMARY、REPRODUCIBILITY_GUIDE，未做项只标 NOT RUN/BLOCKED。

## 预算（不是实测时间）

| Lab | PDF 规定的计算单元 | 估计方法及边界 |
|---|---|---|
| 0 | 13 完整训练 + CUDA 最多 8；E4 ≥10 warmup、每 B ≥30 次 | `13*K*T_epoch`，CUDA另算；K 默认10，可降5；先测真实 CPU epoch。60s/epoch、K10 时 7800s≈2h10min，不含诊断与额外 GPU；这是算术场景，不是本机承诺 |
| 1 | E1 3×深度诊断；E2 3×深度×3epoch + LR grid；E3 6 runs | 每配置单 epoch/诊断计时后求和；默认宽256，CPU慢可128并记录 |
| 2 | LR搜索6 runs；E1 10 runs×10000 steps；E3 新增最多4 runs；CF2 | 先测每架构 step；S固定，batch128；CPU允许小架构/S3000，不因有6GB而擅改标准架构 |
| 3 | 四位置×2000..8000 steps；E3新训练最多4；外推/复杂度 | width128/head4/batch32，测最大训练长度；内存不足仅按PDF记录截断 |
| 4 | E1 SSL100epoch/B1 25；E2完整+3消融25，各边界额外2；E3 probe | N256，梯度累积不等于真实大对比 batch；6GB 先测，必要迁移 Colab；仅无GPU允许宽32/20epoch/E2减两操作 |
| 5 | 网格/≤200epoch；扰动两模式；E2三seed/三模型；深度网格 | 先测200epoch CPU；加入 null 构建、probe和残差边界成本 |
| 6 | EBM5000step×50链；flow四深度各5000；E3六训练 | 方差检查8链×(10000 burn-in+200000)，需单独预算，不能缩成短链冒充 |
| 7 | 两schedule各20000step；E3最多六；稀有模态2次；CF T×10训练 | sampler N5000；DDIM每K和oracle五计时重复；高成本CF另计 |
| 8 | checkpoint推理/温度/PGD | ≥4扰动×6级；FGSM五ε；PGD10与20step/2restart；CPU允许攻击固定1000图 |
| 9 | 一次视觉特征提取+音频CNN；多头/匹配控制/三seed | 先测缓存与头训练；协议与encoder冲突确认前不做正式数据处理 |

## 本地资源门

默认 num_workers=0、CPU 线程2；短测试限制 wall time，逐个运行，记录 RAM/显存/利用率。用户随后明确要求忽略温度监控等待，故不把传感器缺失当成执行阻塞；温度保持未测，不能根据 GPU 低温推断 CPU 低温。OOM 保存失败日志并先分析；只有 PDF 允许的缩减才可标对应变体。没有无人值守通宵计划。

Colab 迁移包包含源码、精确依赖、profile哈希、数据来源/校验、共享split、parent RUN_ID、checkpoint及RNG；先在 Colab 核验数据/版本，另存其环境 provenance，不保证跨硬件逐位一致。
