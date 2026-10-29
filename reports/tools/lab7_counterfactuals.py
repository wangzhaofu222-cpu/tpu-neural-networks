"""Measured counterfactual answers; no inferred result is substituted for a missing RUN."""
from pathlib import Path
import torch
from src.data.gmm2d import target,roles_and_stats,ExactPredictor
from src.diffusion.schedules import make_schedule
from src.diffusion.samplers import ddpm_sample,ddim_sample
from src.diagnostics.trajectory import trajectory_modes
from src.utils.io import write_json
from reports.tools.lab7_common import R,base,read,source_model,evaluate


def main():
    torch.set_num_threads(2)
    e1=read(R/'E1.json');e2=read(R/'E2.json');e3=read(R/'E3.json');d=read(R/'diagnostics.json');s=read(R/'supplements.json')
    linear=e1['models']['linear']['quality'];cosine=e1['models']['cosine']['quality']
    timing=next(r for r in e2['rows'] if r['predictor']=='network' and r['sampler']=='ddpm')
    q1={'question':'T增大十倍，质量是否必然改善？','RUN_IDs':[s['Q1']['quality']['RUN_ID'],timing['RUN_ID']],
        'original_alpha_bar_T':e1['theory']['linear']['alpha_bar_T'],'new_alpha_bar_T':s['Q1']['alpha_bar_T'],
        'original_terminal_KL':e1['theory']['linear']['train_mean_terminal_KL'],'new_terminal_KL':s['Q1']['terminal_KL'],
        'NFE_ratio':s['Q1']['quality']['NFE']/timing['NFE'],
        'actual_median_time_ratio':s['Q1']['quality']['seconds_median']/timing['seconds_median'],
        'original_quality':{k:linear[k] for k in ['MMD2','Dmode','Cov']},
        'new_quality':{k:s['Q1']['quality'][k] for k in ['MMD2','Dmode','Cov']},
        'training_source':s['Q1']['training_source'],'training_updates':20000,
        'explanation':'保持beta范围，增大T降低终点KL并将NFE乘10，但改变训练时间分布及学习问题。质量是否改善分别看实测MMD²、Dmode，不能由终点KL下降推出必然改善；只做一个seed，不声称普适结论。'}
    q2={'question':'零噪声预测器的最终方差','RUN_IDs':[s['Q2']['quality']['RUN_ID']],
        **{k:v for k,v in s['Q2'].items() if k!='quality'},
        'explanation':'从vT=1逐步递推v(t−1)=v(t)/alpha(t)+posterior_var(t)，最后一步只除alpha1。噪声预测项抵消反向过程中的扩张并将质量引向数据；缺失该项得到高方差云，不是目标GMM。方差在标准化坐标比较，5000样本估计有抽样误差。'}
    q3={'question':'较低Lsimple是否保证更好样本？','RUN_IDs':[linear['RUN_ID'],cosine['RUN_ID']],
        'models':{k:{'final_train_Lsimple':e1['models'][k]['training_summary']['train_loss'],
            'fixed_val_Lsimple':e1['models'][k]['training_summary']['validation_loss'],
            'MMD2':e1['models'][k]['quality']['MMD2'],'Dmode':e1['models'][k]['quality']['Dmode'],
            'SNR_less_point01_fraction':e1['theory'][k]['SNR_less_point01_fraction'],
            'E_epsilon_by_t':d[k]['errors']} for k in ['linear','cosine']},
        'explanation':'Lsimple按各自调度均匀抽t，平均的是不同噪声难度的任务；高噪声时epsilon容易预测。需结合Eepsilon随t曲线与SNR<.01比例，按原始尺度MMD²及Dmode评价生成质量。Lsimple不是NLL，两个调度的训练损失不能单独证明质量排序。'}
    q4={'question':'终点网络误差如何被放大？','RUN_IDs':[d[k]['RUN_ID'] for k in ['linear','cosine']],
        'models':{k:{'terminal_error':d[k]['errors'][-1],
            'E_net_T_div_SNR':d[k]['errors'][-1]['E_net']/d[k]['errors'][-1]['SNR']} for k in ['linear','cosine']},
        'explanation':'按式7.7，Enet(T)/SNR(T)是每坐标x0估计相对精确后验均值的MSE，不是epsilon误差本身。先保留实际数值，再对照各调度DDIM K5样本；小SNR会放大很小的网络残差，但中间步骤还能纠正，不能把第一步误差直接等同最终MMD。'}
    # CosineK5 is an additional necessary evaluation forQ4, not a new training condition.
    cosine5=evaluate('Q4_cosine_DDIM5',e1['models']['cosine']['training_source'],'ddim',5)
    linear5=next(r for r in e2['rows'] if r['predictor']=='network' and r['sampler']=='ddim' and r['K']==5)
    q4['DDIM5']={'linear':linear5,'cosine':cosine5};q4['RUN_IDs'] += [linear5['RUN_ID'],cosine5['RUN_ID']]
    candidates=[{'seed':p['seed'],'arm':arm,'source':p['training_source'],'quality':p[arm]} for p in e3['pairs'] for arm in ['A','B'] if p[arm]['Cov']==1]
    if candidates:
        chosen=max(candidates,key=lambda r:r['quality']['Dmode']);q=chosen['quality']
        reference=read(R/'metrics'/f"{base()['data']['pi0']}.json")['reference_frequencies']
        deviations=[a-b for a,b in zip(q['mode_frequencies'],reference)]
        model,cfg,_=source_model(chosen['source']);schedule=make_schedule(cfg['diffusion']).to(cfg['runtime']['device'])
        _,stats=roles_and_stats(cfg);a=e3[chosen['arm']]
        xT=torch.randn(1000,2,generator=torch.Generator().manual_seed(q['initial_seed']),dtype=torch.float64).float().to(cfg['runtime']['device'])
        keep=[0]+[round(f*schedule.T) for f in [.8,.6,.4,.2,.05]]
        if a['sampler']=='ddpm':
            _,info=ddpm_sample(model,schedule,xT,keep=keep,generator=torch.Generator(device=xT.device).manual_seed(q['noise_seed']))
        else:
            # Save all tau states; report nearest retained time if requested time is absent.
            tau=__import__('src.diffusion.samplers',fromlist=['make_tau']).make_tau(schedule.T,a['K'])
            _,info=ddim_sample(model,schedule,xT,a['K'],keep=[0]+tau)
        saved=info['saved'];snapshots={}
        times_used={}
        for requested in keep:
            nearest=min(saved,key=lambda t:abs(t-requested));snapshots[requested]=saved[nearest];times_used[requested]=nearest
        from src.data.gmm2d import standardize,inverse_standardize
        dist=target(cfg);final=dist.mode_id(inverse_standardize(saved[0].double(),stats));centers=standardize(dist.centers,stats)
        modes=[]
        for requested,actual in times_used.items():
            if requested==0:continue
            labels=torch.cdist(saved[actual].double(),schedule.alpha_bar[actual].sqrt().cpu()*centers).argmin(1)
            modes.append({'requested_t':requested,'actual_t':actual,'agreement_all':float((labels==final).double().mean()),
                'per_final_mode':[float((labels[final==m]==m).double().mean()) if (final==m).any() else None for m in range(dist.M)]})
        path=Path('runs/lab7')/q['RUN_ID'];torch.save({'saved':saved,'times_used':times_used},path/'Q5_extra_trajectory.pt')
        q5={'question':'Cov=1是否足够？','RUN_IDs':[q['RUN_ID']],**chosen,
            'reference_frequencies':reference,'frequency_minus_reference':deviations,
            'largest_absolute_deviation_category':max(range(len(deviations)),key=lambda i:abs(deviations[i])),
            'trajectory_mode_agreement':modes,'trajectory_retained_times_used':times_used,
            'explanation':'选择正式E3中Cov=1且Dmode最大的实际运行。Cov只检查每模式频率是否超过参考一半，允许过量或不足；Dmode还计入全部频率与离群类别。轨迹给出同一模型/采样器在不同噪声阶段对最终模式的指向比例，只作描述，不把模式决定归因于单一时间点。DDIM若不含请求时间，明确使用最近保留tau，不声称拥有不存在的中间状态。'}
    else:q5={'question':'Cov=1是否足够？','RUN_IDs':[p[a]['RUN_ID'] for p in e3['pairs'] for a in ['A','B']],
             'status':'BLOCKED','explanation':'正式E3无Cov=1条件，保留全部运行证据，不能虚构教师要求的案例；Cov/Dmode解释另据真实频率给出。'}
    answers=[q1,q2,q3,q4,q5];write_json(R/'counterfactual_answers.json',answers)
    text=['# Lab7 五个反事实问题（实测证据）','']
    for i,answer in enumerate(answers,1):
        text += [f'## Q{i}: '+answer['question'],'',answer['explanation'],'','RUN_ID: '+', '.join(answer['RUN_IDs']),'',
                 '```json',__import__('json').dumps({k:v for k,v in answer.items() if k not in ['explanation','question']},ensure_ascii=False,indent=2),'```','']
    (R/'counterfactual_answers.md').write_text('\n'.join(text),encoding='utf-8')
    print('Five counterfactual answers saved',flush=True)


if __name__=='__main__':main()
