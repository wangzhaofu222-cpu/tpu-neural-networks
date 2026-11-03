# Lab9 实验与证据汇总

实验及证据已完成；教师要求的正式报告尚未制作。

Lab8 的239项测试、827项审计通过，历史结果和checkpoint原样保留。本轮完整回归 304 项测试通过；独立 33305 项检查通过。

个性化配置：实际视觉来源 cnn_lab2，保留冲突字段 vit_small 并标记 UNVERIFIED；scalar门控，p_drop=.5。官方RAVDESS共1440片段、24演员；train840/val180/cal120/test300，演员隔离。cal仅归档完整性校验/身份清单，不解码媒体或生成特征。教师split未能核对，采用用户授权的1729后备划分。mel和池化特征均仅用train拟合标准化。

## E1

|策略|Val宏F1|Test宏F1|准确率|可训练参数|评价RUN_ID|
|---|---:|---:|---:|---:|---|
|video|0.367578|0.288684|0.306667|66696|73b7c031da6acfd09b36-3d018c3486d345a5bfbf442430be428e|
|audio|0.450167|0.438427|0.456667|17544|872fb29a87d782c21051-cf1b54e8d2144cdda6e02712bcaef9bf|
|early|0.441990|0.488125|0.486667|83080|05e36ac62488f7796fd6-370796fdf66a43a0adedf95d30e3e7ec|
|gated|0.486164|0.509938|0.523333|100105|347c6157a20f623bbee4-d363e4726c5f48af848904756452a54e|
|late05|0.453507|0.456888|0.493333|84240|6bce04d107b27fb66fc8-ba4bd30fd0654388bf9658deb27fd3af|
|late_val|0.473582|0.450878|0.476667|84240|e4f5de4987b4b2abe336-1f93deb10d944a25a354bce75dbbc85c|
|video_plus_83080|0.374748|0.292051|0.306667|82847|22f9948b3abb4e7644b2-4373bbba4ee44e4e8a9d1d8257447580|
|audio_plus_83080|0.455090|0.424482|0.440000|83030|747355daa60d2cc08a42-7018b6909b604f41b5dd48f0f459c1ae|
|video_plus_100105|0.375731|0.321880|0.303333|100040|72d3e5c3ceda440965c2-8330b1637b8e41988ef1ee3046ecd911|
|audio_plus_100105|0.451224|0.416379|0.436667|100155|e4b3f5a0c4e93bcc9098-08ab915d587344f9b4b4d15e30657a3e|
|video_plus_84240|0.379016|0.308484|0.310000|84410|8eb4311ca1c7650f5e21-3a6cc9195aa04290b33eeaa1488774f4|
|audio_plus_84240|0.459745|0.423007|0.440000|84263|af77dcffac5fbb7f9ab1-e7e966beb7934d78b0835c8f4079cf83|

主比较 F=gated，U=audio_plus_100105。事前Git：`23f41527e22bd35ed26fa3aa728a99b233fa7878`，差值方向 F−U。三个seed1/2/3均在事前提交后重新训练；探索性seed1已经被观察过，不被描述为全新独立证据。

|seed|F宏F1|U宏F1|差值|F评价RUN_ID|U评价RUN_ID|
|---|---:|---:|---:|---|---|
|1|0.509938445|0.416378607|0.093559838|5eaab7ee33e888c71fa7-fceda1ca709347878fe9090a1126afd6|6d0bc1656002341df4e9-4eda2af5d1be4aeab430cced95c91b87|
|2|0.528176173|0.442624674|0.085551499|071c6c3a878330b2e586-ef72728ed1404818bfb6d9dda97e29db|e680333297ee1350883d-f8690e49d712401fb142404bab7178fa|
|3|0.453179355|0.425341767|0.027837589|a29d0207d0691539b9fb-052d3bc061f04cbca48da4fb1d243237|78d4d7224dc94ed76d19-fe318f92354b4fa9aab2d71e9007a865|

差值均值=0.06898297529200559，样本SD=0.035857223785062814。различие не установлено；科学预测状态=NOT_ESTABLISHED。三差同号，但均值未达到2×样本SD，不能宣称超越参数匹配控制的提升已经确立。

## E2

|模型|完整宏F1|无音频宏F1|无视频宏F1|Ra|Rv|评价RUN_ID|
|---|---:|---:|---:|---:|---:|---|
|early|0.488125|0.243263|0.339932|0.244862|0.148192|da815801667760cba3f0-f653366b50c04783bc48d69e65dc5308|
|gated|0.509938|0.238197|0.352093|0.271742|0.157846|5c2e15819ab203e984c5-dae3fc0aaf3b4cb69f28de679c8d7226|
|late05|0.456888|0.288684|0.438427|0.168204|0.018461|1b42bfe68b7ff9394755-75d0ae54282d486d96b30b596c0bd289|
|late_val|0.450878|0.288684|0.438427|0.162195|0.012452|f72622bc263024125579-3642aca850794e0aae3c4a14fb0a585d|
|gated_dropout|0.439444|0.305552|0.381911|0.133893|0.057533|57166cbd0c7ff3e2e33f-64da840110db4d13a3e90f81c190fba6|

噪声质量低于无音频的最高已测SNR：{"gated": 20, "gated_dropout": 30}。这是离散网格比较，不是连续或单调阈值；dropout曲线非单调，30dB处差异很小。噪声按原始波形真实功率加入，再由相同冻结AudioCNN编码。逐片段SNR/概率/门控见E2.json、corrupt缓存与对应predictions.pt。

缺失模态训练提高了本seed下两种失效状态的F1，但完整输入F1下降；不把单seed方向性结果当作总体显著结论。

## E3

A、B各300个唯一有序配对，均不同演员，seed1。B没有唯一真值，Accuracy/F1/ECE/选择性风险明确未定义并由接口拒绝。

|策略|A宏F1|B跟随视频rv|B跟随音频ra|B第三类r0|AUROC1-conf|AUROCdelta_a|评价RUN_ID|
|---|---:|---:|---:|---:|---:|---:|---|
|early|0.434759|0.173333|0.296667|0.530000|0.501611|0.504500|75cdef3b3621ec713c4d-8f3b5388e56e4af5b0e635d25d018b33|
|gated|0.440716|0.133333|0.340000|0.526667|0.564300|0.556700|bfaf1a95582bbc738bad-8f91e3804e2741cfb08e2df37488ac16|
|late05|0.417586|0.136667|0.436667|0.426667|0.539272|0.590767|281261d23ef5b6ce2337-311678e304db457f8417e218a87b070e|

三种模型的ra均高于rv，但早期/门控约53%的输出属于第三类，不能说所有冲突都跟随音频。AUROC仅是描述性结果，置信度并非可靠冲突判据。

五个反事实问题见counterfactuals.md/json，包含真实双宽控制训练、原始脸帧模糊再编码。g=1表示视频，g=0表示音频；曲线来自实测，不预设门控随SNR的趋势。

## 交付统计和限制

{
  "tests_passed": 304,
  "audit_passed": 33305,
  "requirement_status_counts": {
    "PASS": 47,
    "FAIL": 0,
    "BLOCKED": 0,
    "UNVERIFIED": 2,
    "NOT_APPLICABLE": 2
  },
  "formal_training_runs": 19,
  "evaluation_runs": 41,
  "optimizer_updates": 53262,
  "checkpoint_count": 38,
  "training_wall_seconds": 299.0827747999938,
  "run_wall_seconds_sum": 299.0827747999938,
  "figures": 12,
  "figure_files": 24,
  "preregistration_commit": "23f41527e22bd35ed26fa3aa728a99b233fa7878",
  "note": "Download/preprocessing wall times recorded separately; overlapping durations must not be added as elapsed wall time",
  "measured_session_interval_start": "2026-11-02T20:19:47.943259+07:00",
  "measured_session_interval_end": "2026-11-02T17:03:46.629386+00:00",
  "elapsed_seconds_since_first_recorded_verification": 13438.686127,
  "elapsed_interval_scope": "First recorded Lab8 verification test start -> summary generation, including download/waits/preprocessing/experiments/audit; earlier initial inspection excluded"
}

历史状态原样保留：{"Lab1_E2_growth_bound": "FAIL_RETAINED", "Lab3_E2_raw_memory_intervals": "FAIL_RETAINED", "Lab4_Q2": "BLOCKED", "Lab5_course_split": "UNVERIFIED", "Lab5_E2_magnitude": "FAIL", "Lab6_E1_mixing": "FAIL", "Lab8_E2_positive_prediction": "FAIL; reverse difference established", "Lab8_initial_CPU_GPU_audit": "FAIL_RETAINED; deterministic audit settings corrected, final827PASS"}。

仅5个test演员、一个划分，限制泛化；配对复用源片段，身份、情绪组成与语义不一致存在混杂。视觉编码器从CIFAR迁移至人脸，存在域和输入尺度差异。zero干预依赖具体替代向量，输出依赖不等于有益因果作用。可训练头容量、冻结编码器及late零权重有效容量分别记录。开发异常、容量/时间元数据修复及不改变权重/输入的证明见execution_failures与alignment_metadata_audit.json。未制作正式报告，未启动其他实验。
