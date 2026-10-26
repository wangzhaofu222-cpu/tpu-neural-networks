# Lab0 阶段验收与证据

Lab0必做工程、主实验、八个反事实问题及六节俄语PDF报告已完成。完整课程仍未完成。主实验机器验收见`reports/lab0/acceptance.json`，报告构建与逐页视觉核对见`reports/lab0/build_status.json`。未将附加GPU计时失败或LaTeX编译缺失隐藏在PASS中。

| 范围 | 状态 | 证据 |
|---|---|---|
| 环境/profile/五CLI/公共runner | PASS | src/、configs/、requirements.lock.txt、docs/environment/current.json |
| 数据角色、归一化、完整checkpoint与RNG | PASS | src/data/、src/engine/、25项测试、真实E1/E3 |
| E0 | PASS | loss1.964259→0.895176，val.6232，独立evaluate一致；e0-selected.json |
| E1 A/B/C | PASS | 同模式配对全部step loss一致、accuracy差0；原始ledger |
| E2 | PASS | 预定5seed，mean.633440/std.008708/95%CI[.622628,.644252]；额外seed不回填主CI |
| E3 A/B/C | PASS | 真正进程重启恢复，参数差0、轨迹一致、有效更新数一致；C一次overflow跳步保留 |
| E4主CPU/GPU N100 | PASS | 各50预热/100测量，指定B1/16/64/256，全部主系列CV≤10%，每系列独立RUN_ID |
| 附加GPU N300 CV≤10% | FAIL | B1约20.8%、B16约13.6%；原始样本保留，来源未完全定位，报告明确不保证跨系列latency稳定 |
| 八个CF | PASS | 六节报告第5节；workers2两个真实运行、model-only完整续训与单步分量诊断、同acc不同轨迹配对、同路径1byte干预 |
| 六节俄语PDF及tex源 | PASS | reports/lab0/report.pdf、main.tex、report.ru.md；8页逐页检查 |
| LaTeX编译 | BLOCKED | 内置编译器进程启动1060；完整tex保留，PDF由单独锁定的ReportLab环境导出 |
| 发布配置/比较显示改动 | PASS | 25/25 tests；release-compatibility.json验证同数据及真实首epoch数值一致，未宣称新代码重跑全部正式实验 |

主系列代码固定在`5abedb11663b56690456dca8e8fa3dba24fbb3a7`，E0来源另见其manifest。重新从Git blob计算的逐文件hash与主系列source_hash完全一致，证据为`experiments/lab0/6116b9614f294c4c918500d92471fc4e/source_file_hashes.json`。恢复历史源码时保留LF行尾，关闭Git自动CRLF转换，不能改变字节后仍声称source_hash相同。

教师“13次完整训练”的计数包含复用前半程的resume实验。这里按13条逻辑训练路径完成；完整与半程/续程进程分别有RUN_ID，注册表保留实际epoch和父子关系，不把6epoch中断项伪称12epoch独立训练。GPU B/C各4条逻辑路径另计；三个CF4附加seed另计。首次K10未达loss阈值的记录、预算运行、诊断和fixture均不冒充通过的E0。

原文登记中§7–28的必做工程/实验/报告条款关联本证据；背景、控制题与未选择的扩展任务保留原状态，不把2153个原文段落宣称为全部已完成的实验。
