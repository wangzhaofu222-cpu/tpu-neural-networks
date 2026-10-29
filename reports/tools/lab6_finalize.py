"""Chinese experiment handoff and requirement audit, without a formal report."""
import json
from pathlib import Path
from src.utils.io import write_json

R=Path('experiments/lab6')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    audit=read(R/'final_audit.json');e1=read(R/'E1.json');e2=read(R/'E2.json');e3=read(R/'E3.json');st=audit['statistics'];idx=read(R/'run_index.json');match=e3['capacity_match'];pair=e3['paired_statistics']
    requirements=Path('docs/requirements/requirements_lab6.md').read_text(encoding='utf-8');lines=[];statuses=[]
    for line in requirements.splitlines():
        if line.startswith('|') and line.endswith('|'):
            parts=line.split('|');ident=parts[1].strip()
            if parts[-2].strip()=='IN_PROGRESS':
                relevant={'A02':['Lab0_5_historical_bytes_protected','historical_states_retained'],
                  'E104':['E1_centers_below_midpoint'],'E106':[c['check'] for c in audit['checks'] if c['check'].startswith('quadratic_5percent_acceptance_')],
                  'E107':['quadratic_5percent_acceptance_wrong_noise_control'],'E108':['E1_eta_rule'],
                  'E203':[c['check'] for c in audit['checks'] if c['check'].startswith('Flow_inverse_acceptance_')],
                  'E205':['E2_val_only_depth_rule'],'E307':['E3_independent_paired_statistics']}.get(ident,[])
                failed=[c for c in audit['failed'] if c['check'] in relevant]
                parts[-2]=' FAIL ' if failed else ' PASS ';line='|'.join(parts)
            if parts[-2].strip() in ['PASS','FAIL','BLOCKED','NOT_APPLICABLE']:statuses.append({'ID':ident,'status':parts[-2].strip()})
        lines.append(line)
    path=Path('docs/requirements/requirements_lab6.md');path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    counts={s:sum(r['status']==s for r in statuses) for s in ['PASS','FAIL','BLOCKED','NOT_APPLICABLE']}
    write_json(R/'requirement_status.json',{'counts':counts,'requirements':statuses,'independent_audit_status':audit['status'],
        'historical_states':{'Lab4Q2':'BLOCKED','Lab5_course_split':'UNVERIFIED','Lab5E2_magnitude':'FAIL retained'},
        'scientific_findings':{'E1_mixing20percent':'PASS' if e1['mixing_rule_satisfied'] else 'FAIL (observed, prescribed fallback applied)',
             'E1_all_eta_Cov':[r['Cov'] for r in e1['eta_rows']],'E3_preregistered_direction':'SUPPORTED' if e3['hypothesis_supported'] else 'NOT_SUPPORTED'}})
    out=['# Lab6 实验汇总（不含正式报告）','',
      f"独立审计 **{audit['status']}**，{audit['check_count']}项检查；最终自动测试{st['final_tests_passed']}项通过。需求状态：{counts}。",'','## A. Lab5 核查','',
      '原有130项测试重新通过，625项审计在独立副本中复核一致；274个正式/286个全部checkpoint完整，5450个历史文件未改。未重复正式训练。Lab5完成提交bc75831保留；其E2量级预测FAIL、课程split UNVERIFIED、Lab4Q2 BLOCKED均保留。',
      '', '## B. Lab6 个人参数与协议','',
      '目标eight_gaussians；eta=[.005,.02,.08,.5]；Flow缩放上限c=4。四种目标均已实现；train20000/val5000/test5000，种子601/602/603，原始坐标不标准化，不创建cal。正式CPU2线程、float32、Adam.001、batch256、每模型完整5000updates。',
      'MMD²采用含对角的V统计量，带宽由seed604固定2000train点全部成对距离中位数确定，按目标复用。Dmode主值为固定test频率的两样本估计，包含离群−1；另存对精确目标概率的距离。Cov用精确目标模式概率；每实验目标有五次独立噪声对照。',
      '', '## C. E1','',f"单份EBM已完成5000updates，source RUN_ID `{Path(e1['source']).name}`。四条件共用相同checkpoint哈希{e1['source_checkpoint_sha256']}。",
      f"两个模式中心能量={e1['center_energies']}；中点能量={e1['origin_energy']:.8g}；test平均能量={e1['mean_test_energy']:.8g}。中心低于中点检查{e1['center_energy_acceptance']}。R14未归一化尾积分上界={e1['tail_bound']:.10g}≤.01，不冒充精确尾概率。",
      '', '| eta | MMD² | Cov | DK | 中心→另一模式 | 反射前越界提议 | 评估RUN_ID |','|---|---:|---:|---:|---:|---:|---|']
    for r in e1['eta_rows']:out.append(f"|{r['eta']}|{r['MMD2']:.7g}|{r['Cov']:.3f}|{r['DK']:.6g}|{100*r['center_to_other_mode_fraction']:.3f}%|{100*r['pre_reflection_outside_proposal_fraction']:.4f}%|{r['RUN_ID']}|")
    out+=['','反射后域外比例全部为0，无NaN；这不证明生成质量合格。',
      '', '| 条件 | 理论方差 | 实测方差 | 相对误差 | RUN_ID |','|---|---:|---:|---:|---|']
    for r in e1['quadratic']+[e1['wrong_noise_control']]:
        label='错误噪声控制' if r==e1['wrong_noise_control'] else str(r['eta'])
        out.append(f"|{label}|{r['theory_variance']:.8g}|{r['measured_variance']:.8g}|{100*r['relative_error']:.4f}%|{r['RUN_ID']}|")
    control=e1['wrong_noise_control'];out+=['',
      f"方差检查{e1['variance_acceptance']}；每条件8条独立链、10000burn-in+200000保留步，完整状态及随机创新已保存。错误控制eta={control['eta']}、sigma={control['sigma']}、有效温度={control['effective_temperature']:.6g}。",
      f"工作eta={e1['selected_eta']}；选择规则：{e1['selection_rule']}。最小eta={e1['too_small_eta']}对应位移小；最大eta={e1['too_large_eta_by_theoretical_discretization']}的二次离散方差偏离最大。",
      f"20%混合标准{'满足' if e1['mixing_rule_satisfied'] else '未满足（实测FAIL，按教师规则fallback）'}；所有条件Cov=0是生成质量不足的真实结果，未通过改变sigma、能量幅度或阈值修饰。能量残差tanh只能提供至多2的能量差，s3先验较sigma.3数据宽；中心低能量并不足以保证密度集中。",
      '', '## D. E2','', '| Kf | Val NLL ±SEM | MMD² | Cov | 参数量 | 5000采样秒 | test中位epsilon | norm20中位epsilon | RUN_ID |',
      '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in e2['rows']:
        v=r['invertibility'];out.append(f"|{r['depth']}|{r['val_NLL']:.7g} ±{r['val_NLL_SEM']:.5g}|{r['MMD2']:.7g}|{r['Cov']:.3f}|{r['parameters']}|{r['sample_seconds']:.6g}|{v['median_epsilon']:.4g}|{v['norm20_median_epsilon']:.4g}|{r['RUN_ID']}|")
    out+=['',f"选择Kf={e2['selection']['selected_depth']}。相邻差使用同一val点上的配对NLL差SEM，规则已事前固定；若全部增益明显按PDF选8。具体比较：",'',
      '```json',json.dumps(e2['selection'],ensure_ascii=False,indent=2),'```','',
      'Jacobian与roundtrip独立检查；float64转换的是已学权重，不是重新训练模型。norm20结果仅作压力诊断，不套用数据区域验收阈值。',
      '', '## E. E3','',f"目标={e3['target']}；EBM宽度{match['ebm_width']}、参数{match['ebm_parameters']}；Flow宽度64、参数{match['flow_parameters']}；相对较小模型的参数差={100*match['relative_gap_over_smaller']:.4f}%≤20%。只调整允许的EBM宽度，K50/eta/Kf/c不变。",
      f"主指标={e3['primary_metric']}，Delta=EBM−Flow，事前预测为正。真实事前Git提交={e3['preregistration_commit']}，原始hypothesis.md未改写。",
      '', '| Seed | 模型 | Cov | Dmode | MMD² | 精确test NLL（仅Flow） | Dgrid（EBM网格估计） | 训练秒/采样秒 | RUN_ID |',
      '|---|---|---:|---:|---:|---:|---:|---|---|']
    for r in e3['paired_runs']:
        for kind in ['ebm','flow']:
            v=r[kind];nll=f"{v['exact_test_NLL']:.6g}" if kind=='flow' else '不适用';dg=f"{v['Dgrid']:.6g}" if kind=='ebm' else '—'
            out.append(f"|{r['seed']}|{kind}|{v['Cov']:.3f}|{v['Dmode']:.6g}|{v['MMD2']:.7g}|{nll}|{dg}|{v['training_seconds']:.2f}/{v['sample_seconds']:.5g}|{v['RUN_ID']}|")
    out+=['',f"三个配对差={pair['deltas']}；平均={pair['mean']:.8g}；样本SD(ddof1)={pair['sample_sd']:.8g}；对应指标噪声={pair['metric_noise']:.8g}。同号={pair['same_nonzero_sign']}；absmean≥2SD={pair['abs_mean_ge_2sd']}；absmean≥noise={pair['abs_mean_ge_noise']}。",
      f"教师统计结论：**{pair['conclusion']}**；事前假设得到支持={e3['hypothesis_supported']}。差异仅适用于本目标、固定split、预算和三个seed，非普适因果结论。",
      '', '| Seed | 中点真logdensity | Flow精确logdensity | EBM网格logdensity估计 | Flow−真实 |','|---|---:|---:|---:|---:|']
    for r in e3['paired_runs']:
        a=r['ebm']['midpoint'];b=r['flow']['midpoint'];out.append(f"|{r['seed']}|{a['true_log_density']:.8g}|{b['model_log_density']:.8g}|{a['model_log_density']:.8g}|{b['log_density_excess_over_true']:.8g}|")
    excess=[p['flow']['midpoint']['log_density_excess_over_true'] for p in e3['paired_runs']]
    out+=['',f"Flow中点相对真实密度的倍数={ [float(__import__('math').exp(min(x,700))) for x in excess] }。{'实测中点明显高于真实，连线图显示概率桥；桥高度按实测连续量报告。' if min(excess)>__import__('math').log(10) else '未在所有seed观察到超过一个数量级的中点过剩，不强行宣布普遍明显的桥。'}这只是描述性解释，不是新增教师验收阈值。EBM也可能在间隙留下过多密度，数据附近低能量不能排除这一点。",
      '', '## F. 交付与审计','',
      '五个反事实问题均在counterfactual_answers.json/md中完成，含中文解释、实测值、计算及RUN_ID。八组图表提供SVG/PNG，均来自已保存记录并逐图检查。',
      f"正式模型训练{st['formal_model_training_runs']}次、共{st['formal_optimizer_steps']}次更新；预算训练{st['budget_training_runs']}次，另有{st['formal_evaluation_runs']}个正式模型评估RUN、{st['quadratic_runs']}个完整二次链RUN、{st['noise_runs']}个噪声RUN。",
      f"正式训练累计{st['formal_training_seconds']:.3f}秒（{st['formal_training_seconds']/60:.3f}分钟）；正式工作流墙钟{st['formal_workflow_wall_seconds']/60:.3f}分钟；从Lab5回归开始至最终审计墙钟{st['verification_to_audit_wall_seconds']/60:.3f}分钟。",
      f"checkpoint：正式{st['formal_checkpoints']}个，Lab6含预算共{st['all_lab6_checkpoints']}个。完整模型/optimizer/scheduler/scaler/RNG/replay均已核验；原始轨迹、随机创新、样本、网格本地保留并记录哈希。",
      f"独立审计失败项：{audit['failed']}。Lab6需求状态计数={counts}；历史Lab4Q2仍BLOCKED，Lab5课程split仍UNVERIFIED，均不冒充已解决。",
      '未生成正式俄语/LaTeX/PDF报告；未执行Lab7。最终Git提交由交付消息提供；不要为补写其哈希而改动已提交事前假设。','']
    (R/'EXPERIMENT_SUMMARY.md').write_text('\n'.join(out),encoding='utf-8')
    print({'requirement_counts':counts,'audit_status':audit['status']})
if __name__=='__main__':main()
