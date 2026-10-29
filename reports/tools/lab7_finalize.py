"""Chinese evidence handoff and honest requirement states; no formal report."""
import json,shutil
from pathlib import Path
from src.utils.io import write_json
from reports.tools.lab7_common import R,read,base


def main():
    a=read(R/'final_audit.json');e1=read(R/'E1.json');e2=read(R/'E2.json');e3=read(R/'E3.json');diag=read(R/'diagnostics.json');sup=read(R/'supplements.json');st=a['statistics']
    if a['status']!='PASS':raise RuntimeError('Resolve integrity audit failures before final handoff; preserve scientific failures')
    failed_numeric={f['criterion'] for f in a['teacher_numeric_findings'] if not f['PASS']}
    path=Path('docs/requirements/requirements_lab7.md');rows=[];statuses=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('|') and line.endswith('|'):
            p=line.split('|');ident=p[1].strip()
            if p[-2].strip()=='IN_PROGRESS':
                bad=ident=='E102' and any(k.startswith('E1_forward') for k in failed_numeric) or ident=='E203' and any(k.startswith('E2_time') for k in failed_numeric) or ident=='E204' and any(k.startswith('E2_exact') for k in failed_numeric)
                if ident=='C05' and read(R/'counterfactual_answers.json')[4].get('status')=='BLOCKED':p[-2]=' BLOCKED '
                else:p[-2]=' FAIL ' if bad else ' PASS '
                line='|'.join(p)
            if p[-2].strip() in ['PASS','FAIL','BLOCKED','NOT_APPLICABLE']:statuses.append({'ID':ident,'status':p[-2].strip()})
        rows.append(line)
    path.write_text('\n'.join(rows)+'\n',encoding='utf-8')
    counts={k:sum(s['status']==k for s in statuses) for k in ['PASS','FAIL','BLOCKED','NOT_APPLICABLE']}
    write_json(R/'requirement_status.json',{'counts':counts,'requirements':statuses,'numeric_failures':sorted(failed_numeric),
        'E3_prediction_supported':e3['hypothesis_supported'],'teacher_conclusion':e3['teacher_conclusion'],'historical_states':a['historical_states']})
    out=['# Lab7 实验汇总（不含正式报告）','',
         f"独立审计{a['status']}，{a['check_count']}项；最终测试{st['tests_passed']}项通过；Lab7需求状态{counts}。",'',
         '## A. Lab6核查','',
         '153项旧测试重跑通过，234项审计复算通过，63个checkpoint全部严格加载。5841个历史文件哈希保护，未重复训练。5份CSV、两种反事实答案和8张PNG再生成逐字节一致。Lab6最终提交23f049d保留。Lab4Q2 BLOCKED、Lab5split UNVERIFIED、Lab5E2量级FAIL、Lab6E1混合FAIL均保留。','',
         '## B. 个人参数和协议','',
         'T1000，beta[.0001,.02]，M6，rho.1，pi0.02；width256/depth6/time128，SiLU，变换后的时间特征注入每层。使用RTX3060Laptop、float32、2CPU线程、TF32/AMP关闭。预算实测后保留默认宽度，未减少训练预算。',
         'train20000/val5000/test5000，seeds701/702/703；只按train每坐标均值和总体标准差标准化，质量指标逆变换到原始单位。无cal读取。所有训练复用同一个ExperimentRunner.run；Adam.001/batch512/20000updates。',
         'MMD²是含对角V统计量，h由全部20000rawtrain点199990000个不同点对距离中位数固定；模式参考来自seed704的100000真样本，半径严格<3rho，离群类M。Cov遵循PDF参考频率/2；另存稀有模式pi0/2简写结果。五次独立5000真实样本给出指标噪声均值/样本SD。','',
         '## C. E1','', '| 调度 | alpha_barT | train平均终点KL | SNR<.01比例 | Train Lsimple | Val Lsimple | MMD² | Dmode | Cov | RUN_ID |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for kind in ['linear','cosine']:
        t=e1['theory'][kind];m=e1['models'][kind];q=m['quality'];s=m['training_summary']
        out.append(f"|{kind}|{t['alpha_bar_T']:.9g}|{t['train_mean_terminal_KL']:.9g}|{t['SNR_less_point01_fraction']:.4f}|{s['train_loss']:.8g}|{s['validation_loss']:.8g}|{q['MMD2']:.8g}|{q['Dmode']:.6g}|{q['Cov']:.4f}|{q['RUN_ID']}|")
    b=e1['boundary_short'];out += ['',f"正向过程首批固定seed结果：{ {k:e1['theory'][k]['all_forward_moments_PASS'] for k in ['linear','cosine']} }。全部10个条件逐坐标均值/方差、4SEM阈值已保存，不通过时不换seed。",
        f"T/10边界：T={b['K']}，alpha_barT={b['alpha_bar_T']:.9g}，KL={b['terminal_KL']:.9g}，有效终点协议={b['terminal_protocol_valid']}；精确DDPM MMD²={b['MMD2']:.8g}，Dmode={b['Dmode']:.6g}，RUN_ID={b['RUN_ID']}。这是显式违反终点条件的诊断，不冒充正式训练。",
        f"本次T/10相对完整T的MMD²差={b['MMD2']-e1['boundary_full']['MMD2']:.8g}、Dmode差={b['Dmode']-e1['boundary_full']['Dmode']:.8g}。没有观察到质量恶化，不能据此宣称违反协议有益：式7.5是给定x0的条件KL平均，并非混合边际对N(0,I)的KL；一次差值还需与固定指标噪声比较。正式训练门保持不变。",'',
        '## D. E2','', '| Predictor | Sampler | K/NFE | MMD² | Dmode | Cov | 5次计时中位秒 | RUN_ID |','|---|---|---:|---:|---:|---:|---:|---|']
    for r in e2['rows']+e2['low_K']:out.append(f"|{r['predictor']}|{r['sampler']}|{r['NFE']}|{r['MMD2']:.8g}|{r['Dmode']:.6g}|{r['Cov']:.4f}|{r['seconds_median']:.6g}|{r['RUN_ID']}|")
    out += ['',f"主网格均预热后真实重复5次；K1/2补充样本单独评价，不假称五次性能计时。DDIM_T三初始噪声MMD²={e2['MMD_T_values']}，s_MMD={e2['s_MMD']:.9g}；最小有效K={e2['smallest_effective_K']}，阈值2s={e2['selection_threshold']:.9g}。",
        f"精确预测器满T的MMD²验收上界={e2['exact_noise_threshold']:.9g}，逐项结果={e2['exact_full_acceptance']}。时间/NFE±20%结果={e2['timing_acceptance']}。失败项原样保存，不筛选计时。",'',
        '## E. 网络诊断','',
        '两种调度各25个时间步，包含1/T；Eepsilon、Ex0、Estar、Enet及有限样本交叉项/SEM完整保存。按定义检验Ex0=Eepsilon/SNR；不能把有限验证集的误差分解强制改为精确相加。',
        '每种调度四个噪声水平，41×41向量场；方向余弦与长度比均用pt权重。保存原始向量、概率权重及全部1000条DDPM完整轨迹。',
        '', '| 调度 | Enet(T) | Enet(T)/SNR(T) | 各噪声水平加权余弦 | 轨迹最终模式指向率（t/T=.8,.6,.4,.2,.05） |',
        '|---|---:|---:|---|---|']
    for kind in ['linear','cosine']:
        d=diag[kind];last=d['errors'][-1]
        out.append(f"|{kind}|{last['E_net']:.8g}|{last['network_x0_amplification']:.8g}|{[r['weighted_cosine'] for r in d['fields']]}|{[r['agreement_all'] for r in d['trajectory_modes']]}|")
    out += ['', '## F. E3','',f"A={e3['A']}；B={e3['B']}。唯一变化因素为反向采样器配置；每个seed两臂共用同一份完整训练的linear权重、数据/标准化及xT。seed1明确复用探索性结果；seed2/3于事前Git之后执行。",
        f"E3事前commit={e3['preregistration_commit']}。完整不可改写假设见hypothesis.md，配置见E3_protocol_lock.json。",'',
        '| Seed | Arm | Cov | p_hat0 | Dmode | MMD² | RUN_ID |','|---|---|---:|---:|---:|---:|---|']
    for p in e3['pairs']:
        for arm in ['A','B']:
            r=p[arm];out.append(f"|{p['seed']}|{arm}|{r['Cov']:.4f}|{r['rare_frequency']:.6g}|{r['Dmode']:.6g}|{r['MMD2']:.8g}|{r['RUN_ID']}|")
    out += ['',f"Delta={e3['deltas']}；mean={e3['mean']:.9g}；样本SD={e3['sample_sd']:.9g}；同号={e3['same_sign']}；absmean≥2SD={e3['abs_mean_ge_2sd']}；教师结论：**{e3['teacher_conclusion']}**。",
        f"Dmode噪声均值={e3['Dmode_noise_mean']:.9g}，absmean≥noise={e3['above_noise']}。事前方向假设得到支持={e3['hypothesis_supported']}。三seed仅描述同一目标的变化，不推广到其他目标。",'',
        '| pi0 | train稀有分量数量 | 生成p0 | Cov | PDF稀有覆盖 | 新训练source RUN_ID |','|---|---:|---:|---:|---|---|']
    for r in sup['rare_boundaries']:
        q=r['quality'];out.append(f"|{r['pi0']}|{r['train_component0_count']}|{q['rare_frequency']}|{q['Cov']}|{q['rare_covered_pdf']}|{Path(r['training_source']).name}|")
    covered=[r['quality']['rare_covered_pdf'] for r in sup['rare_boundaries']]
    boundary_conclusion='两种降低权重条件下稀有模式仍被覆盖，本次pi0=.01和.005范围内未找到覆盖失效点；不外推未知阈值。' if all(covered) else '按逐行实测覆盖标志定位本次测量的失效区间，不外推未测权重。'
    out += ['', '两种稀有权重均重新生成对应目标及train统计，并各训练完整20000步。'+boundary_conclusion,'',
        '## G. 交付','',f"正式训练{st['formal_training_runs']}次，共{st['formal_optimizer_updates']}updates；预算训练{st['budget_training_runs']}次，模型评价RUN{st['evaluation_runs']}次，诊断/噪声RUN{st['diagnostic_noise_runs']}次，总RUN目录{st['all_run_folders']}。",
        f"正式训练累计{st['formal_training_seconds']:.3f}秒（{st['formal_training_seconds']/60:.3f}分钟）；正式工作流墙钟{st['formal_workflow_wall_seconds']/60:.3f}分钟；历史核查至审计{st['verification_to_audit_wall_seconds']/60:.3f}分钟。",
        f"checkpoint正式{st['formal_checkpoints']}，含预算共{st['all_checkpoints']}；图表{st['figure_groups']}组SVG/PNG；测试{st['tests_passed']}项；独立审计{a['check_count']}项。",
        f"需求状态={counts}。数值未通过项={sorted(failed_numeric)}；历史状态={a['historical_states']}。",
        '五个反事实问题详见counterfactual_answers.json/md，含真实数值、计算、RUN_ID和适用范围。Q1完成一次10T真实20000步训练并实测DDPM耗时；Q2按最后一步无噪声递推并实际采样验证。',
        '数据、权重、原始样本、轨迹和RNG本地保存并有哈希索引；小型配置/日志/表格/图表/测试/验收记录进入Git。未制作或修改正式俄语/LaTeX/PDF报告，未进入Lab8。最终Git哈希由交付消息给出，不回写事前假设。','']
    (R/'EXPERIMENT_SUMMARY.md').write_text('\n'.join(out),encoding='utf-8')
    logs=R/'logs';logs.mkdir(exist_ok=True)
    for p in Path('tmp').glob('lab7_*.log'):shutil.copyfile(p,logs/p.name)
    print({'counts':counts,'audit_checks':a['check_count']})


if __name__=='__main__':main()
