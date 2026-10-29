"""Tables, scientific plots and measured counterfactuals from immutable Lab6 RUNs."""
import csv,json,math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from src.utils.io import write_json,file_hash
from src.generative.distributions import make_roles

R=Path('experiments/lab6');F=R/'figures'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(run,name):return torch.load(Path('runs/lab6')/run/name,weights_only=False,map_location='cpu')
def table(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def save(fig,name,caption):
    fig.suptitle('Lab6 · original coordinates · fixed data seeds601/602/603',fontsize=11)
    fig.tight_layout(rect=(0,.04,1,.95));fig.text(.5,.008,caption,ha='center',fontsize=8)
    fig.savefig(F/(name+'.svg'));fig.savefig(F/(name+'.png'),dpi=160);plt.close(fig)
def density(ax,array,title,segment=None,points=None):
    a=np.asarray(array).reshape(120,120);positive=np.maximum(a,1e-12)
    image=ax.imshow(positive,origin='lower',extent=(-15,15,-15,15),norm=LogNorm(vmin=1e-6,vmax=max(float(positive.max()),1e-5)),cmap='viridis')
    if points is not None:ax.scatter(points[:,0],points[:,1],s=1,c='white',alpha=.3)
    if segment is not None:
        centers=np.asarray(segment);ax.plot(centers[:,0],centers[:,1],'o-',c='#D55E00',lw=2,markersize=4)
    ax.set(title=title,xlabel='x1',ylabel='x2',xlim=(-15,15),ylim=(-15,15));return image

def e1_figures(e1,idx):
    rows=e1['eta_rows'];table('E1_eta.csv',rows);table('E1_quadratic.csv',e1['quadratic']+[e1['wrong_noise_control']])
    selected=idx['evaluations'][f"E1_eta_{e1['selected_eta']}"];grid=load(selected['RUN_ID'],'density_grid.pt');data=make_roles('two_blobs')['train'][:500].numpy()
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    im=ax[0].imshow(grid['energy'].reshape(120,120),origin='lower',extent=(-15,15,-15,15),cmap='magma');ax[0].scatter(data[:,0],data[:,1],s=1,c='white');ax[0].set(title='Learned EBM energy',xlabel='x1',ylabel='x2');fig.colorbar(im,ax=ax[0],shrink=.7)
    im=density(ax[1],grid['model_log_density'].exp(),'EBM grid-normalized density',points=data);fig.colorbar(im,ax=ax[1],shrink=.7)
    im=density(ax[2],grid['true_log_density'].exp(),'True two_blobs density',points=data);fig.colorbar(im,ax=ax[2],shrink=.7)
    save(fig,'E1_density','EBM normalization is estimated on Omega=[-15,15]^2; it is not exact likelihood. White points: fixed first500train samples.')
    fig,ax=plt.subplots(2,3,figsize=(12,7));eta=[r['eta'] for r in rows]
    specs=[('MMD2','MMD²',1),('Cov','Mode coverage (%)',100),('DK','Mean displacement DK',1),
           ('center_to_other_mode_fraction','Center-to-other-mode endpoint (%)',100),('outside_final_fraction','Final outside Omega (%)',100),
           ('pre_reflection_outside_proposal_fraction','Outside proposals before reflection (%)',100)]
    for a,(key,label,scale) in zip(ax.flat,specs):
        a.plot(eta,[scale*r[key] for r in rows],'o-',color='#0072B2');a.set(xscale='log',xlabel='eta (log scale)',ylabel=label)
        a.axvline(e1['selected_eta'],ls=':',c='#D55E00',label='Selected eta')
        if key=='center_to_other_mode_fraction':a.axhline(20,ls='--',c='gray',label='Required20%')
    ax[0,0].legend(fontsize=8);ax[1,0].legend(fontsize=8)
    ax[0,1].set_ylim(0,100);ax[1,1].set_ylim(0,5.5)
    save(fig,'E1_eta','One frozen EBM; K50,T1,5000generated samples percondition. Endpoint mixing and boundary proposal metrics are distinct.')
    q=e1['quadratic']+[e1['wrong_noise_control']];labels=[str(r['eta']) for r in q[:-1]]+['wrong-noise']
    fig,ax=plt.subplots(1,2,figsize=(10,4));xx=np.arange(len(q))
    ax[0].plot(xx,[r['theory_variance'] for r in q],'s--',c='#D55E00',label='Finite-step theory')
    ax[0].plot(xx,[r['measured_variance'] for r in q],'o-',c='#0072B2',label='Measured');ax[0].set(yscale='log',ylabel='Stationary variance',xticks=xx,xticklabels=labels);ax[0].legend()
    ax[1].bar(xx,[100*r['relative_error'] for r in q],color='#0072B2');ax[1].axhline(5,ls='--',c='#D55E00',label='5% acceptance');ax[1].set(ylabel='Relative variance error (%)',xticks=xx,xticklabels=labels);ax[1].legend()
    save(fig,'E1_quadratic','Each condition:8independent chains,10000burn-in+200000retained updates, no boundary; full finite trajectories saved.')
    fig,ax=plt.subplots(1,4,figsize=(13,3.7))
    for a,r in zip(ax,rows):
        traj=load(r['RUN_ID'],'sampling_trajectory.pt')[:,:32].numpy()
        for j in range(32):a.plot(traj[:,j,0],traj[:,j,1],lw=.5,alpha=.5,c='#0072B2')
        a.scatter(traj[0,:,0],traj[0,:,1],s=8,c='black',marker='x');a.scatter(traj[-1,:,0],traj[-1,:,1],s=8,c='#D55E00')
        a.set(title=f"eta={r['eta']}",xlabel='x1',ylabel='x2',xlim=(-15,15),ylim=(-15,15))
    save(fig,'E1_trajectories','Fixed first32chains, all51states; black x=initial, orange=final. No chain selected by its outcome.')

def e2_figures(e2):
    rows=[]
    for r in e2['rows']:
        row={k:v for k,v in r.items() if k!='invertibility'};row.update(r['invertibility']);rows.append(row)
    table('E2_depths.csv',rows)
    fig,ax=plt.subplots(1,4,figsize=(13,4));x=[r['depth'] for r in rows]
    ax[0].errorbar(x,[r['val_NLL'] for r in rows],yerr=[r['val_NLL_SEM'] for r in rows],fmt='o-',capsize=4,c='#0072B2');ax[0].set_ylabel('Val exact NLL ±1SEM')
    ax[1].plot(x,[r['MMD2'] for r in rows],'s-',c='#D55E00');ax[1].set_ylabel('MMD² (5000samples)')
    ax[2].plot(x,[r['median_epsilon'] for r in rows],'o-',c='#0072B2',label='Test median')
    ax[2].plot(x,[r['norm20_median_epsilon'] for r in rows],'s--',c='#D55E00',label='Norm20 median');ax[2].axhline(1e-5,c='gray',ls=':');ax[2].set(yscale='log',ylabel='Float64 relative inverse error');ax[2].legend(fontsize=7)
    ax[3].plot(x,[r['sample_seconds'] for r in rows],'^-',c='#009E73');ax[3].set_ylabel('Generate5000samples (seconds)')
    for a in ax:a.set(xscale='log',xticks=x,xticklabels=x,xlabel='Coupling depth Kf')
    save(fig,'E2_depth','Each model5000updates; depth selected using paired val-NLL difference SEM. Norm20 is a separate100point stress diagnostic.')

def e3_figures(e3):
    rows=[]
    for pair in e3['paired_runs']:
        for kind in ['ebm','flow']:
            r=pair[kind];row={k:r.get(k) for k in ['RUN_ID','source_RUN_ID','seed','parameters','training_seconds','sample_seconds','Cov','Dmode','MMD2','exact_test_NLL','Dgrid','Z_hat_omega','normalized_on_omega','tail_unnormalized_integral_upper_bound_R14']}
            row.update(kind=kind,midpoint_true_log_density=r['midpoint']['true_log_density'],midpoint_model_log_density=r['midpoint']['model_log_density'],midpoint_density_kind=r['midpoint']['model_density_kind']);rows.append(row)
    table('E3_models.csv',rows)
    paired=e3['paired_statistics'];table('E3_pairs.csv',[{'seed':r['seed'],'EBM':r['ebm'][e3['primary_metric']],'Flow':r['flow'][e3['primary_metric']],'delta':d,'EBM_RUN_ID':r['ebm']['RUN_ID'],'Flow_RUN_ID':r['flow']['RUN_ID']} for r,d in zip(e3['paired_runs'],paired['deltas'])])
    fig,ax=plt.subplots(2,3,figsize=(11,7))
    for j,pair in enumerate(e3['paired_runs']):
        for i,kind in enumerate(['ebm','flow']):
            r=pair[kind];s=load(r['RUN_ID'],'generated_samples.pt').numpy();outside=int((np.abs(s)>15).any(1).sum())
            ax[i,j].scatter(s[:,0],s[:,1],s=1,alpha=.3,c='#0072B2' if kind=='ebm' else '#D55E00')
            ax[i,j].set(title=f"{kind},seed{pair['seed']},outsideOmega={outside}",xlabel='x1',ylabel='x2',xlim=(-15,15),ylim=(-15,15))
    save(fig,'E3_samples','All5000generated samples perRUN; displayed areaOmega. Any samples outside the displayed area are counted in each panel title.')
    first=e3['paired_runs'][0];ebm=first['ebm'];flow=first['flow'];eg=load(ebm['RUN_ID'],'density_grid.pt');fg=load(flow['RUN_ID'],'density_grid.pt');segment=ebm['midpoint']['centers']
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    for a,data,title in [(ax[0],eg['true_log_density'].exp(),'True target'),(ax[1],eg['model_log_density'].exp(),'EBM Omega-grid estimate'),(ax[2],fg['model_log_density'].exp(),'Flow exact density')]:
        im=density(a,data,title,segment);fig.colorbar(im,ax=a,shrink=.7)
    save(fig,'E3_density','Seed1maps; orange segment connects nearest modes. Display floor1e-12 only; stored metrics use original densities.')
    fig,ax=plt.subplots(1,2,figsize=(11,4));base=load(ebm['RUN_ID'],'bridge_line.pt');fraction=np.linspace(0,1,201)
    ax[0].plot(fraction,base['true_log_density'],'k--',label='True target')
    for pair in e3['paired_runs']:
        for kind,style,c in [('ebm','-','#0072B2'),('flow',':','#D55E00')]:
            line=load(pair[kind]['RUN_ID'],'bridge_line.pt');ax[0].plot(fraction,line['model_log_density'],ls=style,c=c,alpha=.7,label=f"{kind},seed{pair['seed']}")
    ax[0].set(xlabel='Fraction along nearest-center segment',ylabel='Log density');ax[0].legend(fontsize=7)
    ax[1].scatter([1,2,3],paired['deltas'],s=45,c='#0072B2',label='Paired EBM−Flow')
    ax[1].axhline(0,c='black',lw=1);ax[1].errorbar([3.6],[paired['mean']],yerr=[paired['sample_sd']],fmt='D',capsize=4,c='#D55E00',label='Mean ±sampleSD')
    ax[1].set(xlabel='Seed / mean at right',ylabel=f"Delta {e3['primary_metric']}",xticks=[1,2,3]);ax[1].legend(fontsize=8)
    save(fig,'E3_bridge_pairs','EBM logdensity uses estimated gridZ; Flow density is exact. Primary metric and difference order were fixed before E3.')

def counterfactuals(e1,e2,e3):
    small=e1['eta_rows'][:2];large=e1['eta_rows'][-1];q=e1['quadratic'][-1];ctrl=e1['wrong_noise_control'];depths=e2['rows'][:2]
    mids=[{'seed':p['seed'],'EBM_RUN_ID':p['ebm']['RUN_ID'],'Flow_RUN_ID':p['flow']['RUN_ID'],'true':p['ebm']['midpoint']['true_log_density'],
       'EBM_grid_estimate':p['ebm']['midpoint']['model_log_density'],'Flow_exact':p['flow']['midpoint']['model_log_density']} for p in e3['paired_runs']]
    answers=[
      {'question':'Q1:固定K下eta趋零会怎样？','measurements':small,'explanation':f"eta={small[0]['eta']}时DK={small[0]['DK']:.6g}，eta={small[1]['eta']}时DK={small[1]['DK']:.6g}。固定K=50时，漂移总尺度随eta、累计噪声尺度随sqrt(K*eta)减小，链趋向初始点；小步长并不保证短链样本接近目标。"},
      {'question':'Q2:最大eta的理想二次链和学习能量首先在哪方面偏离？','quadratic':q,'learned':large,
       'explanation':f"二次无边界链理论方差={q['theory_variance']:.6g}，实测={q['measured_variance']:.6g}，误差={100*q['relative_error']:.3f}%。学习能量最大步长的stable={large['stable']}、Cov={large['Cov']:.6g}、MMD²={large['MMD2']:.6g}；四条件完整曲线记录于E1_eta.csv。稳定性仅说明没有数值发散，不能说明覆盖或样本质量正确。二次关系只约束其自身模型的方差，不预测学习能量的Cov/MMD²。"},
      {'question':'Q3:错误噪声匹配对应什么平稳规律？','measurements':ctrl,
       'explanation':f"eta={ctrl['eta']}，固定sigma={ctrl['sigma']:.6g}，Teff={ctrl['effective_temperature']:.6g}；离散平稳方差={ctrl['measured_variance']:.6g}，理论sigma²/(2eta−eta²)={ctrl['theory_variance']:.6g}。连续过程的理想二次分布方差是Teff，有限步长离散链多出1/(1−eta/2)因子，不能混为一谈。"},
      {'question':'Q4:更好的likelihood是否意味着更好的生成样本？','measurements':depths,
       'explanation':f"深度{depths[0]['depth']}：valNLL={depths[0]['val_NLL']:.6g}，MMD²={depths[0]['MMD2']:.6g}；深度{depths[1]['depth']}：valNLL={depths[1]['val_NLL']:.6g}，MMD²={depths[1]['MMD2']:.6g}。这对实测中NLL与MMD排序{'一致' if (depths[0]['val_NLL']-depths[1]['val_NLL'])*(depths[0]['MMD2']-depths[1]['MMD2'])>0 else '不一致'}。即使此次一致也不是普遍蕴含关系：NLL评价对数据点密度，MMD评价生成样本分布，且有限样本有噪声。"},
      {'question':'Q5:数据附近低能量是否证明模式间密度低？','measurements':mids,
       'explanation':'不能。模式中点的三个logdensity及整条连线已实测；EBM项明确为Omega网格归一化估计，Flow项为精确密度。数据能量低没有单独约束中点能量、归一化常数或生成链覆盖。目标高斯混合的中点密度很小但非零，支持仍是整个平面；实测桥不能单独证明严格断连支持的拓扑限制。结合有限深度、基础分布及优化讨论桥强度，不预设桥必然明显，也不把所有差异解释成拓扑的单一因果作用。'}]
    write_json(R/'counterfactual_answers.json',answers)
    text=['# Lab6 五个反事实问题（实测分析，不是正式报告）','']
    for a in answers:
        text+=['## '+a['question'],'',a['explanation'],'','实测值、计算与 RUN_ID：','', '```json',json.dumps({k:v for k,v in a.items() if k not in ['question','explanation']},ensure_ascii=False,indent=2),'```','']
    (R/'counterfactual_answers.md').write_text('\n'.join(text),encoding='utf-8')

def main():
    F.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    idx=read(R/'run_index.json');e1=read(R/'E1.json') if (R/'E1.json').exists() else None;e2=read(R/'E2.json') if (R/'E2.json').exists() else None;e3=read(R/'E3.json') if (R/'E3.json').exists() else None
    if e1:e1_figures(e1,idx)
    if e2:e2_figures(e2)
    if e3:e3_figures(e3);counterfactuals(e1,e2,e3)
    write_json(R/'figure_provenance.json',{'generator':'reports.tools.lab6_artifacts','input_hashes':{p.name:file_hash(p) for p in [R/'E1.json',R/'E2.json',R/'E3.json',R/'run_index.json'] if p.exists()},
        'figure_hashes':{p.name:file_hash(p) for p in F.iterdir() if p.is_file()},'RUN_ID_mapping':'E1_eta.csv,E1_quadratic.csv,E2_depths.csv,E3_models.csv,E3_pairs.csv'})
    print('Artifacts regenerated; PNG count=',len(list(F.glob('*.png'))))
if __name__=='__main__':main()
