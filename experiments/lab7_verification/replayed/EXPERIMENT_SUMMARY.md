# Lab7 实验汇总（不含正式报告）

独立审计PASS，467项；最终测试180项通过；Lab7需求状态{'PASS': 47, 'FAIL': 0, 'BLOCKED': 0, 'NOT_APPLICABLE': 4}。

## A. Lab6核查

153项旧测试重跑通过，234项审计复算通过，63个checkpoint全部严格加载。5841个历史文件哈希保护，未重复训练。5份CSV、两种反事实答案和8张PNG再生成逐字节一致。Lab6最终提交23f049d保留。Lab4Q2 BLOCKED、Lab5split UNVERIFIED、Lab5E2量级FAIL、Lab6E1混合FAIL均保留。

## B. 个人参数和协议

T1000，beta[.0001,.02]，M6，rho.1，pi0.02；width256/depth6/time128，SiLU，变换后的时间特征注入每层。使用RTX3060Laptop、float32、2CPU线程、TF32/AMP关闭。预算实测后保留默认宽度，未减少训练预算。
train20000/val5000/test5000，seeds701/702/703；只按train每坐标均值和总体标准差标准化，质量指标逆变换到原始单位。无cal读取。所有训练复用同一个ExperimentRunner.run；Adam.001/batch512/20000updates。
MMD²是含对角V统计量，h由全部20000rawtrain点199990000个不同点对距离中位数固定；模式参考来自seed704的100000真样本，半径严格<3rho，离群类M。Cov遵循PDF参考频率/2；另存稀有模式pi0/2简写结果。五次独立5000真实样本给出指标噪声均值/样本SD。

## C. E1

| 调度 | alpha_barT | train平均终点KL | SNR<.01比例 | Train Lsimple | Val Lsimple | MMD² | Dmode | Cov | RUN_ID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|linear|4.03582977e-05|4.03591121e-05|0.3260|0.40611092|0.40820262|0.0021704942|0.05625|1.0000|6166b1c66f39bc269e25-5caf87cfcdea4065acccb5837a8371c4|
|cosine|2.42876691e-09|2.42876691e-09|0.0640|0.69308062|0.68685859|0.0021916993|0.05185|1.0000|2db3dda8cd907b076db0-f602c1d1c4e643fdb23d079baf92c928|

正向过程首批固定seed结果：{'linear': True, 'cosine': True}。全部10个条件逐坐标均值/方差、4SEM阈值已保存，不通过时不换seed。
T/10边界：T=100，alpha_barT=0.363563248，KL=0.451870234，有效终点协议=False；精确DDPM MMD²=7.2387351e-05，Dmode=0.01229，RUN_ID=3500712e1f9546c4fd3a-5b44110051804e37ac2ed9be2c34f09d。这是显式违反终点条件的诊断，不冒充正式训练。
本次T/10相对完整T的MMD²差=-6.7465606e-05、Dmode差=-0.00398。没有观察到质量恶化，不能据此宣称违反协议有益：式7.5是给定x0的条件KL平均，并非混合边际对N(0,I)的KL；一次差值还需与固定指标噪声比较。正式训练门保持不变。

## D. E2

| Predictor | Sampler | K/NFE | MMD² | Dmode | Cov | 5次计时中位秒 | RUN_ID |
|---|---|---:|---:|---:|---:|---:|---|
|network|ddpm|1000|0.0021704942|0.05625|1.0000|1.61582|deed2ad27ea6dc9b82a2-9bde9515fc6f4be4b836698669d64ef6|
|network|ddim|5|0.051622578|0.66303|0.1667|0.0089891|8bb864eb20b7936a8893-0029cd3498bd4a598bf5fe81e3e428f5|
|network|ddim|10|0.014722897|0.21945|1.0000|0.0174917|9db76aabb6e8383c4b82-084d6fec26054359b64c0b8ddc79a3a4|
|network|ddim|20|0.008565276|0.12585|1.0000|0.0407095|4f2b4ed83734347bcc82-3bde0c5c0f674beead091bc398cc0951|
|network|ddim|50|0.0066820911|0.09245|1.0000|0.0861675|a1bee33a693a2f0588bb-59d9b1c17aab43d9b3a85c3b6e9aa76d|
|network|ddim|100|0.0060812626|0.08865|1.0000|0.179459|29aec0ea2f756acc6639-1de816295c7a472d9cdbd69b2d37f74a|
|network|ddim|1000|0.0056763812|0.08525|1.0000|1.82389|0fbd4ffcb74fc35a520b-b72ba38fc00d4ee1991e8673c276ae66|
|exact|ddpm|1000|0.00013985296|0.01627|1.0000|2.05166|2fe25d54fee39b604917-ef345080701c4b2895cdc6b24f7c10c7|
|exact|ddim|5|0.014008456|0.48883|0.5000|0.0118186|622bdc881ca8055f1b01-d6ac5cef7b8d4357b9aa03ce48cb4f7e|
|exact|ddim|10|0.0010714302|0.12103|1.0000|0.0234965|5fa6c52f7d8adf496252-e8beab3114bc446b922737116479b82b|
|exact|ddim|20|0.00024589817|0.01882|1.0000|0.0472515|a875b6acfd5af99091f2-8bcb24336fa6418c8bbfd1ff3757cf1e|
|exact|ddim|50|8.0184062e-05|0.01152|1.0000|0.115042|0f83221e00b224e1a376-0bba1a3a83584af2b79bd4ecf27b6ef1|
|exact|ddim|100|5.0658267e-05|0.01292|1.0000|0.242087|df097c5a7198a8b3c28a-30d04405ad0e44aeaba848cd03631704|
|exact|ddim|1000|3.9539251e-05|0.01169|1.0000|2.37442|44630ed4fab7291adb68-9bbb419ea8b040778ef10fa9e5c40405|
|network|ddim|1|0.57330362|0.98063|0.0000|0.0025772|66c8b48041c5283a85d1-efd774427a9944d7891b459215d5091b|
|network|ddim|2|0.1284461|0.98563|0.0000|0.0044307|251b7fcad9e3cd7d22ed-61674e24f2594f1f804bea5acc1f02df|
|exact|ddim|1|0.13034998|0.98903|0.0000|0.0031448|fcb808b60bd0ce223fc2-702c82c1f041441782ffdc4f89af2ca8|
|exact|ddim|2|0.10797491|0.98863|0.0000|0.0058105|e32a84e7e0024abc2a2e-0283fb99577d4db49c7f51f90a73d0ee|

主网格均预热后真实重复5次；K1/2补充样本单独评价，不假称五次性能计时。DDIM_T三初始噪声MMD²=[0.005676381175473422, 0.006164184842205156, 0.004929828939303826]，s_MMD=0.000621681477；最小有效K=50，阈值2s=0.00124336295。
精确预测器满T的MMD²验收上界=0.000245054983，逐项结果=[{'RUN_ID': '2fe25d54fee39b604917-ef345080701c4b2895cdc6b24f7c10c7', 'PASS': True}, {'RUN_ID': '44630ed4fab7291adb68-9bbb419ea8b040778ef10fa9e5c40405', 'PASS': True}]。时间/NFE±20%结果={'network': {'median_seconds_per_NFE': 0.0017491700000391575, 'conditions': [{'RUN_ID': 'deed2ad27ea6dc9b82a2-9bde9515fc6f4be4b836698669d64ef6', 'relative_deviation': 0.07623472849194213, 'PASS': True}, {'RUN_ID': '9db76aabb6e8383c4b82-084d6fec26054359b64c0b8ddc79a3a4', 'relative_deviation': 0.0, 'PASS': True}, {'RUN_ID': '4f2b4ed83734347bcc82-3bde0c5c0f674beead091bc398cc0951', 'relative_deviation': 0.16368048846156458, 'PASS': True}, {'RUN_ID': 'a1bee33a693a2f0588bb-59d9b1c17aab43d9b3a85c3b6e9aa76d', 'relative_deviation': 0.014761286828624764, 'PASS': True}, {'RUN_ID': '29aec0ea2f756acc6639-1de816295c7a472d9cdbd69b2d37f74a', 'relative_deviation': 0.02596660129468642, 'PASS': True}, {'RUN_ID': '0fbd4ffcb74fc35a520b-b72ba38fc00d4ee1991e8673c276ae66', 'relative_deviation': 0.04271797478756967, 'PASS': True}]}, 'exact': {'median_seconds_per_NFE': 0.0023496499998145737, 'conditions': [{'RUN_ID': '2fe25d54fee39b604917-ef345080701c4b2895cdc6b24f7c10c7', 'relative_deviation': 0.12682471850538035, 'PASS': True}, {'RUN_ID': '5fa6c52f7d8adf496252-e8beab3114bc446b922737116479b82b', 'relative_deviation': 0.0, 'PASS': True}, {'RUN_ID': 'a875b6acfd5af99091f2-8bcb24336fa6418c8bbfd1ff3757cf1e', 'relative_deviation': 0.005500819353060571, 'PASS': True}, {'RUN_ID': '0f83221e00b224e1a376-0bba1a3a83584af2b79bd4ecf27b6ef1', 'relative_deviation': 0.02077075301434339, 'PASS': True}, {'RUN_ID': 'df097c5a7198a8b3c28a-30d04405ad0e44aeaba848cd03631704', 'relative_deviation': 0.030310046261948553, 'PASS': True}, {'RUN_ID': '44630ed4fab7291adb68-9bbb419ea8b040778ef10fa9e5c40405', 'relative_deviation': 0.010542719209036155, 'PASS': True}]}}。失败项原样保存，不筛选计时。

## E. 网络诊断

两种调度各25个时间步，包含1/T；Eepsilon、Ex0、Estar、Enet及有限样本交叉项/SEM完整保存。按定义检验Ex0=Eepsilon/SNR；不能把有限验证集的误差分解强制改为精确相加。
每种调度四个噪声水平，41×41向量场；方向余弦与长度比均用pt权重。保存原始向量、概率权重及全部1000条DDPM完整轨迹。

| 调度 | Enet(T) | Enet(T)/SNR(T) | 各噪声水平加权余弦 | 轨迹最终模式指向率（t/T=.8,.6,.4,.2,.05） |
|---|---:|---:|---|---|
|linear|0.0020229494|50.122722|[0.9791375994682312, 0.9673614501953125, 0.9966871738433838, 0.9951449632644653]|[0.155, 0.197, 0.336, 0.67, 0.989]|
|cosine|0.0025147432|1035399.2|[0.9742338061332703, 0.9864611625671387, 0.9850978255271912, 0.9874650239944458]|[0.285, 0.416, 0.667, 0.943, 0.991]|

## F. E3

A={'schedule': 'linear', 'sampler': 'ddpm', 'K': 1000, 'eta': None}；B={'schedule': 'linear', 'sampler': 'ddim', 'K': 50, 'eta': 0.0}。唯一变化因素为反向采样器配置；每个seed两臂共用同一份完整训练的linear权重、数据/标准化及xT。seed1明确复用探索性结果；seed2/3于事前Git之后执行。
E3事前commit=f65a5ff1e4eb61ce9c387c5a3c75926beaca0f55。完整不可改写假设见hypothesis.md，配置见E3_protocol_lock.json。

| Seed | Arm | Cov | p_hat0 | Dmode | MMD² | RUN_ID |
|---|---|---:|---:|---:|---:|---|
|1|A|1.0000|0.0238|0.05625|0.0021704942|deed2ad27ea6dc9b82a2-9bde9515fc6f4be4b836698669d64ef6|
|1|B|1.0000|0.032|0.09245|0.0066820911|a1bee33a693a2f0588bb-59d9b1c17aab43d9b3a85c3b6e9aa76d|
|2|A|1.0000|0.0176|0.02143|0.00049002328|250c19ef7ca7b9338b63-4f723f99f0904780a4b7afeb083bb9b8|
|2|B|1.0000|0.0146|0.06917|0.0015029847|7312b330f611f76c8e21-a0ad8fecfe1a44d68d0c19e4c31292db|
|3|A|1.0000|0.03|0.02502|0.00040347014|e9dc77c09db6e8a2e9e9-d3abdfee6d00443d82c09e1339d720ea|
|3|B|1.0000|0.0308|0.07214|0.0014280241|c96d001bc559c4c4d5e3-3577f31765e5485e8f0eb3eae5e874c3|

Delta=[-0.036199999999999996, -0.047740000000000005, -0.04712000000000002]；mean=-0.0436866667；样本SD=0.00649105025；同号=True；absmean≥2SD=True；教师结论：**различие установлено**。
Dmode噪声均值=0.009434，absmean≥noise=True。事前方向假设得到支持=True。三seed仅描述同一目标的变化，不推广到其他目标。

| pi0 | train稀有分量数量 | 生成p0 | Cov | PDF稀有覆盖 | 新训练source RUN_ID |
|---|---:|---:|---:|---|---|
|0.01|199|0.0102|1.0|True|9cae2d3989a3456192ed-7ba06c33f8604aa79a42feef9557f7b3|
|0.005|113|0.005|1.0|True|5bb265455cfc0ad6f17d-e83cd88eba9b48a6917e82b49a5d6956|

两种稀有权重均重新生成对应目标及train统计，并各训练完整20000步。两种降低权重条件下稀有模式仍被覆盖，本次pi0=.01和.005范围内未找到覆盖失效点；不外推未知阈值。

## G. 交付

正式训练7次，共140000updates；预算训练2次，模型评价RUN35次，诊断/噪声RUN18次，总RUN目录62。
正式训练累计1924.478秒（32.075分钟）；正式工作流墙钟42.995分钟；历史核查至审计54.118分钟。
checkpoint正式35，含预算共39；图表14组SVG/PNG；测试180项；独立审计467项。
需求状态={'PASS': 47, 'FAIL': 0, 'BLOCKED': 0, 'NOT_APPLICABLE': 4}。数值未通过项=[]；历史状态={'Lab4Q2': 'BLOCKED', 'Lab5_course_split': 'UNVERIFIED', 'Lab5_E2_magnitude': 'FAIL', 'Lab6_E1_mixing': 'FAIL'}。
五个反事实问题详见counterfactual_answers.json/md，含真实数值、计算、RUN_ID和适用范围。Q1完成一次10T真实20000步训练并实测DDPM耗时；Q2按最后一步无噪声递推并实际采样验证。
数据、权重、原始样本、轨迹和RNG本地保存并有哈希索引；小型配置/日志/表格/图表/测试/验收记录进入Git。未制作或修改正式俄语/LaTeX/PDF报告，未进入Lab8。最终Git哈希由交付消息给出，不回写事前假设。
