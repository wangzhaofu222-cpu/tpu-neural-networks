# 逐项需求与验收矩阵

状态：PASS 必须有证据；FAIL 为已验证不通过；BLOCKED 为资源/确认缺失；NOT RUN 为尚未执行。当前全文原文登记 2153 条（含背景、理论、参考文献），见各 Lab clauses 文件及 `clause_register.json`；下表给出必做工作的可实施分组。原文段落 ID 是不可省略的细化追踪，不能只按下表粗粒度打勾。

| ID | PDF 章节 / 页 | 必做内容与验收 | 当前状态 |
|---|---|---|---|
| L0-PRO | §7–11 p4–9 | 五 CLI；SHA256 profile；Python≥3.11/PyTorch2；精确全依赖 lock/hash | NOT RUN |
| L0-RUN | §14–16 p11–13 | Runner全生命周期；manifest所有字段；独立RUN_ID；小文本Git版本化、权重忽略 | NOT RUN |
| L0-DAT | §13 p10–11 | CIFAR43000/5000/2000/10000；复用split；内容与split fingerprints | NOT RUN |
| L0-CKP | §17 p13 | model/optimizer/scheduler/epoch/step/Python/NumPy/Torch/CUDA RNG/AMP；另存loader generator，真正进程重启 | NOT RUN |
| L0-LOG | §18 p13–14 | train/val loss/metric、lr、epoch/step/time；E1逐step loss与first batch checksum | NOT RUN |
| L0-PRF | §19/22.5 p14/18 | 参数、forward/batch latency、throughput、epoch time、GPU peak；10warmup/30测量/均值中位p95/CV≤10%或解释 | NOT RUN |
| L0-CMP | §20 p14 | 汇总所有run、按EXP_ID分组、config diff；相同条件diff为空 | NOT RUN |
| L0-E0 | §21.1/22.1 p15–17 | final train loss≤first/2；CIFAR val acc≥.30；完整日志；独立evaluate差≤1e-6 | NOT RUN |
| L0-E1 | §21.2/22.2 p15/17 | A exact轨迹/结果；B step rel loss≤1e-5、acc≤.001；C实测解释；split/checksum记录冲突见missing_resources | NOT RUN |
| L0-E2 | §21.3/22.3 p15–17 | ≥5seed；mean/样本std/95%CI；最小效应参考2std；std>.015须解释 | NOT RUN |
| L0-E3 | §21.4/22.4 p16–17 | delta_theta A≤1e-7/B≤1e-5；A acc差0/B≤.001；optimizer steps完全相同 | NOT RUN |
| L0-E4 | §21.5/22.5 p16/18 | profile指定全部B、latency/throughput/memory曲线，保留cold首步供CF7 | NOT RUN |
| L0-CF01..08 | §24 p19 | 最小效应/样本数；workers2两运行；只权重resume；近同acc轨迹；cuDNN变化；改数据1字节；cold/warm；最优B差异 | NOT RUN |
| L0-T01..11 | §25 p19–20 | config；split；checkpoint outputs；metric；EXP变化；EXP稳定/RUN变化；不覆盖；代码变化；profile确定/区分；双profile无需改代码；CPU/CUDA配置 | NOT RUN |
| L0-REP | §27–28 p20–21 | 6节：profile预算、harness、E0–4、R0–3带模式、8CF、限制 | NOT RUN |
| L1-IMP | §5–8 p2–8 | 显式He/Xavier/Gaussian；ReLU/tanh、BN/LN、残差；float64 hook统计q/v/zero/saturation/delta/gl/GI/AVD/rho_T/NaN；probe真实B≥64、单位u | NOT RUN |
| L1-E1 | §9.1 p8–9 | 3init×profile深度、宽256/noNorm/ReLU；最大深度chi前/反误差≤20%；排除q<1e-30；Gaussian训练1epoch后复查理论 | NOT RUN |
| L1-E2 | §9.2 p9 | plain/res alpha1/alpha1√L，3epoch；只为最浅plain选LR固定网格；signal增长误差≤20%；scaled q ratio≤1.2e^c；首个rho≥.9/NaN深度 | NOT RUN |
| L1-E3 | §9.3 p10 | Git先假设；一因素A/B三seed；三差值/2std；二维trainability map | NOT RUN |
| L1-CF01..05 | §10 p10–11 | 深256预测；宽×4实测；gl与delta；残差爆炸；近同acc不同GI下个深度 | NOT RUN |
| L1-T01..05 | §11 p11 | hook不改output/grad；权重方差误差≤10%(fan产品≥1e4)；零分支identity≤1e-6；NaN/Inf flag | NOT RUN |
| L1-REP | §12–13 p11 | 6节，全部RUN_ID与5CF | NOT RUN |
| L2-IMP | §5–8 p2–8 | CIFAR ResNet18 stem3/s1无pool；ViT P4/D384/L6/h6/MLP1536；features接口；MAC计数含MHA；嵌套分层fractions；A0/A1/A2；6种扰动参数表 | NOT RUN |
| L2-E1 | §9.1 p9 | S10000/B128；AdamW/LR三点各模型val选；10训练；参数±20%或MAC匹配；MAC误差≤1%；所有fractions图/r* | NOT RUN |
| L2-E2 | §9.2 p10 | r1 checkpoint、不重训；profile三扰动s0..3；δ/均值/leader；clean完全相同；换leader两差幅≥.01 | NOT RUN |
| L2-E3 | §9.3 p10–11 | val选主要模式；Git假设；三共seed/同子集；map+三差值 | NOT RUN |
| L2-CF01..05 | §10 p11 | 数据曲线；少数据同epoch步数重训；MAC/time；P减半测MAC；translation2/4/8额外评估 | NOT RUN |
| L2-T01..06 | §11 p11 | patch roundtrip；logits/features/不改参数；单convMAC；子集嵌套均衡/恰S步；A0确定；扰动s0/标签/seed | NOT RUN |
| L2-REP | §12–13 p12 | 6节含资源/curve/shift/E3 map/5CF限制；保留r1 checkpoint供8/9 | NOT RUN |
| L3-IMP | §5–8 p2–8 | no-pos/absolute P1/P2/P3/sinusoidal/RoPE；d128/h4/FF2d/preNorm/noCausal；64词表/三个生成器/角色seed；显式attention诊断 | NOT RUN |
| L3-E1 | §9.1 p8–9 | B32/S2000→4000→8000，val规则统一S；LR只val L0选；test≥200序列/L0,2,4,8倍；P1额外2倍；no-pos≤1/64+.05、最好领先≥.20或诊断 | NOT RUN |
| L3-E2 | §9.2 p9 | B4/≥5长度64..1024；forward/train-step/实测peak；10warmup；log斜率/最大两点；二次α1.6..2.4、无显式矩阵memory .8..1.2或分解解释 | NOT RUN |
| L3-E3 | §9.3 p9–10 | 两位置方案、一因素、L*>L0；Git先假设、三seeds | NOT RUN |
| L3-CF01..05 | §10 p10 | sinus ER；L×4 memory；P1/P2；RoPE全curve；宽×2计时 | NOT RUN |
| L3-T01..06 | §11 p11 | attention行和1/屏蔽0；无pos等变<1e-5；sin外推；RoPE相对内积1e-5；P2原长1e-6；生成器唯一key/needle/长度不改train | NOT RUN |
| L3-REP | §12–13 p11 | 6节；L0/2L0逐head entropy/distance/target mass，不把热图当解释 | NOT RUN |
| L4-IMP | §5–8 p2–9 | CIFAR ResNet18、512→512→128无bias头；自写稳定NT-Xent；独立views RNG；freeze完整BN；probe/kNN200/τ.07；5000train诊断/trace+rank/近零n/a | NOT RUN |
| L4-E1 | §9.1 p9–10 | N256/SSL100epoch/B1监督25epoch；random/B1/SSL统一probe+kNN；≥5checkpoints含起点；loss≤ln511或查错 | NOT RUN |
| L4-E2 | §9.2 p10 | base+三leave-one-out各25epoch；diff仅aug；半G0规则+collapse；再只留最佳操作/无操作两个边界训练 | NOT RUN |
| L4-E3 | §9.3 p10–11 | 同已训练encoder/cache、三标注子集/probe seeds×fractions×3encoder；Git；主要rmin；明确不是方法重训方差 | NOT RUN |
| L4-CF01..05 | §10 p11 | N64短训练；协议4.6颜色任务；rank vs质量；z_hat probe；B1小标注公平重训 | BLOCKED（CF2，其余NOT RUN） |
| L4-T01..06 | §11 p11–12 | views含workers；known-loss/τ.01有限；解析grad1e-5；BN冻结/kNN仅train；rank1/高维/常数n/a/trace恒等；嵌套/cal报错 | NOT RUN |
| L4-REP | §12–13 p12 | 6节含seed来源/val选择偏差；保留encoder与cache | NOT RUN |
| L5-IMP | §6–8 p6–9 | 课程Planetoid masks、图清理；MLP/GCN/第二GNN；宽64/drop.5/Adam≤200；lr/wd2×2；度/分量/hE/h0/hdeg；删除/添加/null/ER；S/Dirichlet/probe | BLOCKED（课程masks） |
| L5-E1 | §9.1 p9 | 三模型；GCN p0,.1,.3,.5,.8,1训练/只评价两模式；isolates预测误差≤20%；hE保持、Gtop | NOT RUN |
| L5-E2 | §9.2 p10 | **主比较**Git先假设；三seed original/null/MLP；Q10；degree exact、hE/hdeg±20%；ER seed1边界 | NOT RUN |
| L5-E3 | §9.3 p10–11 | 无训练传播与深度profile；rho/GI/Ehat/probe诊断，不混淆不可训练/过拟合/平滑；2std(val E2)；最大深度residual复训 | NOT RUN |
| L5-CF01..05 | §10 p11 | 第二GNN null；加.5E边训练；删边两模式；S/probe；test选深度仅事后泄漏演示不用于正式选择 | NOT RUN |
| L5-T01..06 | §11 p11–12 | MLP无图；空GCN=MLP1e-6；无向无loop/duplicates且p端点；null保degree/success≥.5；200次传播/能量尺度；改test标签不改训练/选择 | NOT RUN |
| L5-REP | §12–13 p12 | 6节含graph map、ER、5CF | NOT RUN |
| L6-IMP | §6–9 p2–9 | 四指定GMM；601/602/603、20k/5k/5k不标准化；Ω[-15,15]²/120²grid；EBM quadratic+tanh；反射Langevin/replay10k；自写coupling/swap/logdet | NOT RUN |
| L6-E1 | §10.1 p9–10 | two_blobs EBM5000step/K50；ηprofile；8链200k+10k burn-in每η；错误σ控制；variance±5%；tail≤.01；两center能量低于0；20%跨模或如实未混合 | NOT RUN |
| L6-E2 | §10.2 p10–11 | Kf2,4,8,16各5000step；NLL val/SE、MMD/Cov/time/params；float64 epsinv median≤1e-5；norm20；按2SE饱和选深度 | NOT RUN |
| L6-E3 | §10.3 p11–12 | profile难目标、参数±20%、三seed；Git先假设；N5000；MMD/Dmode噪声5次；精确NLL与Ω密度分列；三log bridge | NOT RUN |
| L6-CF01..05 | §11 p12 | 小η位移、大ηvariance、错误温度、NLL vs MMD、bridge三log | NOT RUN |
| L6-T01..06 | §12 p12 | scalar/lower bound；σ²2ηT/方差长链；sampler不改θ/negative detach梯度；inverse及独立logdet；identity/swap/点密度batch不变；Ω flag/replay复原 | NOT RUN |
| L6-REP | §13–14 p12–13 | **5节**，不可套6节模板 | NOT RUN |
| L7-IMP | §5–8 p2–10 | rare GMM/train标准化；float64 linear/cosine；αbar0=1/终值≤1e-3；MLP256×6/time128逐层；Adam lr1e-3/B512/20k；exact score；自写DDPM/DDIM | NOT RUN |
| L7-DIAG | §8.4 p9–10 | ≥20t含端点 errors；41²网格≥4噪声pt加权；1000trajectories五切片；5000samples/真实mode频率100k reference/指标噪声5次 | NOT RUN |
| L7-E1 | §9.1 p10–11 | 两schedule等训练；q_sample 10k五t均值/方差4SE；DDPM品质；T/10不改β oracle边界（允许单独诊断不走正式schedule gate） | NOT RUN |
| L7-E2 | §9.2 p11–12 | DDIM K5/10/20/50/100/T vs DDPM；network/oracle；5次计时+warmup；NFE实计；oracle≤noise+3std；K1/2边界 | NOT RUN |
| L7-E3 | §9.3 p12 | Git先假设/一因素/三seed、Dmode/Cov/rare频率；rare weight÷2/÷4再训边界 | NOT RUN |
| L7-CF01..05 | §10 p12–13 | T×10重新训练；zero网络方差；loss与品质；Enet/SNR；Cov1但Dmode差/轨迹 | NOT RUN |
| L7-T01..06 | §11 p13 | schedule；q_sample二维/图像broadcast；x0恢复1e-8/score有限差1e-4；DDIMη1=DDPM1e-6/no-grad/NFE/end；正τ有序；time embeddings | NOT RUN |
| L7-REP | §12–13 p13–14 | 6节，生成图标schedule/sampler/K/η | NOT RUN |
| L8-IMP | §6–8 p6–9 | 复用checkpoint/logits；val拟合logT；ECE15等宽/NLL/Brier；≥4扰动0..5；MSP/H/margin；自写FGSM/PGD；risk/AURC；OOD AUROC/AUPR/FPR95；reliability.json全部6字段 | NOT RUN |
| L8-E1 | §9.1 p9 | 两模型clean前后；argmax100%一致；val NLL不增；M5/50/等频边界 | NOT RUN |
| L8-E2 | §9.2 p9–10 | **主比较**profile架构clean/扰动5温度收益，三checkpoint（允许单checkpoint按分析单位分三test并明确）；Git先假设；level0≤1e-6；T5 val shift边界 | NOT RUN |
| L8-E3 | §9.3 p10–11 | 两模型同subset，FGSMε0/1/2/4/8；PGDprofileε K10/R1/α2.5εK；PGD acc≤FGSM；K20/R2边界 | NOT RUN |
| L8-CF01..05 | §10 p11 | ECE vs NLL/Brier；T5 transfer；等Linf随机扰动；攻击温度；AURC/AUROC温度 | NOT RUN |
| L8-T01..06 | §11 p11 | softmax/argmax；100k校准样本ECE .01；角色/OOD拒算；扰动identity；每PGD步投影/参数不变；zeroCov None/oracleAURC/单调AUROC | NOT RUN |
| L8-REP | §12–13 p12 | 6节；semanticOOD不算acc/ECE/NLL/Brier；nearOOD附加依协议 | NOT RUN |
| L9-PRO | §6–7 p7 | 同时记录两encoder字段；用户授权以visual_encoder来源优先；课程actor split14/3/2/5和840/180/120/300 clips | BLOCKED（数据协议；profile已生成） |
| L9-IMP | §8 p8–9 | video5fps/face_found；audio16kHz/mel64/nfft400/hop160/centerTrue；train音频CNN后冻结；mask显式；整数半开窗；early/late/gate/参数控制；fusion_profile/undefined | BLOCKED（数据/encoder） |
| L9-E1 | §9.1 p9–10 | 单模态/三融合/λ.5+val网格；参数匹配±1%或最近宽解释；**主比较Git先假设，三seed**；G/Gint/Gctrl/confusion | NOT RUN |
| L9-E2 | §9.2 p10–11 | gate modalitydrop profile；零替代三presence；R/Rnorm与shuffle derangement；SNR30/20/10/0/-5；gate趋势及noise比missing差边界 | NOT RUN |
| L9-E3 | §9.3 p11 | 不同演员同标签/异标签各≥200对；F1仅同标签；异标签rv/ra/r0/delta/conf/gate；mismatch AUROC两scores | NOT RUN |
| L9-CF01..05 | §10 p11–12 | 弱模态贡献；单模态控制宽×2重训；shuffle vs mask；video blur；gate/SNR | NOT RUN |
| L9-T01..06 | §11 p12 | clip/actor无泄漏；窗覆盖/pad；missing对随机替换不变；gate/late端点；audio-independent贡献0/derangement；SNR±.1dB/level0/冲突拒算 | NOT RUN |
| L9-REP | §12–13 p12–13 | 6节，冲突异标签acc/F1/ECE/risk必须undefined | NOT RUN |

## 附加任务（默认 NOT RUN，不阻塞必做验收）

- L0-OPT01..05，§29 p21–22：更强dataset fingerprint、registry、sweep、monitoring、lineage。其中内容hash/监控部分为用户额外要求采用，不改变教师验收含义。
- L1-OPT01..03，§14 p12：norm消融、优化器、Jacobian。
- L2-OPT01..03，§14 p12–13：A0/A2训练、教师脸数据迁移、CKA。
- L3-OPT01..03，§14 p12：local attention、依赖距离、attention干预。
- L4-OPT01..03，§14 p13：τ网格、去negative、脸SSL。
- L5-OPT01..03，§14 p13：SBM、脸landmark图、GAT审计。
- L6-OPT01..03，§15 p13–14：链长、OOD likelihood、图像生成。
- L7-OPT01..03，§14 p14：x0/v参数化、图像/CFG、与Lab6比较。
- L8-OPT01..03，§14 p12–13：conformal（首次读cal）、near/far OOD比较、ensemble。
- L9-OPT01..04，§14 p13：cross-attention、missing校准、时间偏移、回归。

所有控制问题与文献保留在原文条款登记；正文 `Что сдать` 要求的是反事实答案，控制问题不冒充需要实测的反事实答案。
