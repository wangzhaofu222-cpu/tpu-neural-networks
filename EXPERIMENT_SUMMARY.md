# 实验项目阶段汇总

Phase A/B已完成；Lab0阶段验收：**PASS**。整个Lab0–Lab9课程项目尚未完成，不能作为全课程完成提交。

## 已完成

- 10份教师PDF、152页全文审计、2153个原文条款ID；原PDF哈希不变。2153包含理论和参考文献，不等于2153项必做实验。
- 项目独立Python3.11.17 / PyTorch2.7.1+cu126 / torchvision0.22.1+cu126，精确锁定依赖；CPU与RTX3060 Laptop CUDA实测可用。
- 公共runner、五CLI、哈希profile、数据角色隔离、完整checkpoint/RNG/GradScaler、逐step日志、独立evaluate、比较与性能测量。
- 自动测试25/25通过，errors=0、failures=0、skipped=0，20.400秒。证据：`docs/environment/tests-lab0-release.xml`。合成fixture测试仅验证实现。

## Lab0真实结果

- 真实CPU预算24.09秒/epoch，12epoch；13个逻辑训练条件估计62.63分钟。主serial-suite实际65.05分钟；此前E0/失败尝试、后续CF4补充与注册profiling另计。
- E0 loss 1.964259→0.895176，val accuracy=0.6232，独立evaluate完全吻合。首次K10的loss比值0.500332未通过，保留FAIL记录；K12从头重训通过。
- A/CPU、B/CUDA、C/CUDA+AMP：E1逐step loss与最终指标重复一致；E3完整进程重启恢复后参数相对差均为0。有效optimizer steps A/B为2016，C为2015；C实际有1次overflow跳步。
- E2预定5seed：mean=0.633440，样本std=0.008708，95% Student t CI=[0.622628,0.644252]；2std参考效应=0.017416。
- workers2配对与workers0基准的全部2016个step losses完全相同。只恢复权重的最终参数相对差=0.371028，不能当成正确恢复。
- CF4额外3个seed单独登记，不重算预定5seed统计；数据1字节干预和optimizer/lr/RNG诊断均有真实记录。
- E4主CPU/GPU系列各50预热/100测量，全部B的CV≤10%；附加GPU N300诊断在B1/B16得到CV约20.8%/13.6%，阈值状态FAIL，已保留并在报告披露，不能把主系列通过扩展为普遍延迟稳定保证。
- 俄语六节报告：`reports/lab0/report.pdf`、`main.tex`、`report.ru.md`；图表、RUN_ID注册表、实验验收及构建状态同目录。
- 主E1/E2/E3与补充训练固定提交`5abedb11663b56690456dca8e8fa3dba24fbb3a7`。发布时显式split参数及比较显示的改动另有兼容验证；不改写历史manifest。

## 课程状态

| Lab | 实现/正式实验/报告 | 限制 |
|---|---|---|
| 0 | PASS | PDF经渲染检查；LaTeX编译器启动1060，tex编译未验证 |
| 1–3 | NOT RUN；需求/profile已登记 | 按依赖次序逐个开发、测试、预算后运行 |
| 4 | NOT RUN；CF2完整验收BLOCKED | 缺课程协议§4.6颜色控制集 |
| 5 | NOT RUN；正式masks核验BLOCKED | 缺课程Planetoid masks |
| 6–7 | NOT RUN；需求/profile已登记 | 长链、训练、采样与CF须各自先做真实预算 |
| 8 | NOT RUN | 缺Lab2训练checkpoint；近OOD附加缺11类名单 |
| 9 | BLOCKED（正式数据） | 缺人脸帧/音频预处理包、演员/clip划分；编码器已按用户授权选cnn_lab2 |

## 已知边界

课程数据协议缺失，自建split不声称逐索引等同教师文件。CIFAR官方页未列独立许可证，不分发原数据。CPU温度未测，用户要求不等待传感器。断点恢复保证仅限完整epoch边界和记录的代码/环境。没有遗留后台或通宵训练计划。LaTeX编译未验证与PDF渲染验收分开记录。
