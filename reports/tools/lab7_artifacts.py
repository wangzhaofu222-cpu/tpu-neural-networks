"""Plots/tables derived exclusively from saved Lab7 records; never trains or samples."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.io import write_json,file_hash

R=Path('experiments/lab7');F=R/'figures'
plt.rcParams.update({'font.size':10,'figure.dpi':120,'svg.hashsalt':'lab7-r4','axes.spines.top':False,'axes.spines.right':False})


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(fig,name):
    fig.tight_layout()
    fig.savefig(F/(name+'.png'),dpi=150)
    fig.savefig(F/(name+'.svg'),metadata={'Date':None})
    plt.close(fig)
def table(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (R/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def pts(run):return torch.load(Path('runs/lab7')/run/'generated_samples.pt',weights_only=False).numpy()


def main():
    F.mkdir(parents=True,exist_ok=True)
    e1=read(R/'E1.json');e2=read(R/'E2.json');diag=read(R/'diagnostics.json')
    e3=read(R/'E3.json') if (R/'E3.json').exists() else None
    sup=read(R/'supplements.json') if (R/'supplements.json').exists() else None
    data=torch.load('data/lab7/base/test.pt',weights_only=False).numpy()
    theory=e1['theory'];fig,ax=plt.subplots(1,3,figsize=(12,3.5))
    for kind in ['linear','cosine']:
        p=Path('runs/lab7')/theory[kind]['RUN_ID'];s=torch.load(p/(kind+'_schedule.pt'),weights_only=False)
        for a,key in zip(ax,['beta','alpha_bar','log_snr']):a.plot(np.arange(1,len(s[key]))/ (len(s[key])-1),s[key][1:],label=kind);a.set(xlabel='t/T',ylabel=key)
    ax[1].set_yscale('log');ax[0].legend();save(fig,'E1_schedules')
    source=Path('runs/lab7')/theory['linear']['RUN_ID'];clouds=torch.load(source/'forward_clouds_full_train.pt',weights_only=False)
    fig,axes=plt.subplots(2,5,figsize=(14,5.8))
    for row,kind in enumerate(['linear','cosine']):
        for a,m in zip(axes[row],theory[kind]['forward_moments']):
            x=clouds[kind+'_'+str(m['t'])].numpy();a.scatter(x[:,0],x[:,1],s=.15,alpha=.25,rasterized=True)
            a.set(title=f"{kind}, t={m['t']}",xlim=(-4,4),ylim=(-4,4),aspect='equal')
    save(fig,'E1_forward_clouds')
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for a,kind in zip(axes,['linear','cosine']):
        path=Path(e1['models'][kind]['training_source']);rows=[json.loads(l) for l in (path/'metrics.jsonl').read_text().splitlines()]
        for key,label in [('train_loss','train'),('validation_loss','fixed val')]:a.plot([r['global_step'] for r in rows],[r[key] for r in rows],label=label,linewidth=.8)
        a.set(title=kind,xlabel='Optimizer updates',ylabel='Lsimple (sum of 2 coordinates)');a.legend()
    save(fig,'E1_training')
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    for a,(kind,x) in zip(axes,[('target',data)]+[(k,pts(e1['models'][k]['quality']['RUN_ID'])) for k in ['linear','cosine']]):
        a.scatter(data[:,0],data[:,1],s=.7,c='gray',alpha=.2,rasterized=True);a.scatter(x[:,0],x[:,1],s=1,alpha=.4,rasterized=True)
        a.set(title=kind if kind=='target' else kind+' / DDPM / K=1000 / eta=n/a',xlim=(-1.7,1.7),ylim=(-1.7,1.7),aspect='equal')
    save(fig,'E1_samples')
    table('E1_models.csv',[{'schedule':kind,**{k:v for k,v in theory[kind].items() if k!='forward_moments'},
        'train_Lsimple':e1['models'][kind]['training_summary']['train_loss'],'val_Lsimple':e1['models'][kind]['training_summary']['validation_loss'],
        **e1['models'][kind]['quality']} for kind in ['linear','cosine']])
    table('E1_forward_moments.csv',[{'schedule':k,**r} for k in ['linear','cosine'] for r in theory[k]['forward_moments']])
    table('E2_sampling.csv',e2['rows']+e2['low_K']+e2['additional_initial_noises'])
    fig,axes=plt.subplots(1,3,figsize=(12,3.8));noise=read(R/'metrics/0.02.json')
    for pred,color in [('network','tab:blue'),('exact','tab:orange')]:
        rows=sorted([r for r in e2['rows'] if r['predictor']==pred and r['sampler']=='ddim'],key=lambda r:r['K'])
        for a,key in zip(axes,['MMD2','Dmode','seconds_median']):
            a.plot([r['NFE'] for r in rows],[r[key] for r in rows],'o-',label=pred+' DDIM eta=0',color=color)
            r=next(r for r in e2['rows'] if r['predictor']==pred and r['sampler']=='ddpm')
            a.scatter(r['NFE'],r[key],marker='*',s=110,color=color,label=pred+' DDPM')
            a.set(xscale='log',xlabel='NFE per trajectory',ylabel={'MMD2':'MMD²','Dmode':'Dmode','seconds_median':'Median sampling time (s)'}[key])
    axes[0].axhline(e2['exact_noise_threshold'],ls=':',c='black',label='noise mean + 3SD')
    axes[1].axhline(noise['Dmode_noise_mean'],ls=':',c='black');axes[0].legend(fontsize=7)
    axes[0].set_yscale('log');axes[2].set_yscale('log');save(fig,'E2_cost_quality')
    fig,axes=plt.subplots(2,2,figsize=(8,7))
    for a,r in zip(axes.flat,e2['low_K']):
        x=pts(r['RUN_ID']);outside=np.mean(np.any(np.abs(x)>2,axis=1))
        a.scatter(data[:,0],data[:,1],s=.7,c='gray',alpha=.15,rasterized=True);a.scatter(x[:,0],x[:,1],s=1,alpha=.3,rasterized=True)
        a.set(title=f"{r['predictor']} DDIM K={r['K']}, eta=0\noutside view={100*outside:.1f}%, Cov={r['Cov']:.2f}",xlim=(-2,2),ylim=(-2,2),aspect='equal')
    save(fig,'E2_extreme_K')
    table('timestep_errors.csv',[{'schedule':k,**r,'RUN_ID':diag[k]['RUN_ID']} for k in ['linear','cosine'] for r in diag[k]['errors']])
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for a,kind in zip(axes,['linear','cosine']):
        rows=diag[kind]['errors']
        for key in ['E_epsilon','E_x0','E_epsilon_star','E_net']:a.plot([r['t'] for r in rows],[max(r[key],1e-15) for r in rows],label=key)
        a.set(title=kind,xlabel='t',ylabel='Per-coordinate MSE',yscale='log');a.legend(fontsize=8)
    save(fig,'diagnostic_timestep')
    for kind in ['linear','cosine']:
        fields=torch.load(Path('runs/lab7')/diag[kind]['RUN_ID']/'score_fields.pt',weights_only=False)
        fig,axes=plt.subplots(4,3,figsize=(10,12))
        for row,f in enumerate(fields):
            x=f['points'].numpy();step=np.arange(0,len(x),3)
            for col,key in enumerate(['network','exact']):
                v=f[key].numpy();u=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
                axes[row,col].quiver(x[step,0],x[step,1],u[step,0],u[step,1],scale=30,width=.002)
                axes[row,col].set(title=f"{kind} t={f['t']} {key} direction",aspect='equal')
            image=axes[row,2].imshow(f['cosine'].reshape(41,41).numpy().T,origin='lower',extent=(-3.5,3.5,-3.5,3.5),vmin=-1,vmax=1,cmap='coolwarm')
            axes[row,2].set(title=f"weighted cos={f['weighted_cosine']:.3f}\nlength ratio={f['weighted_length_ratio']:.3f}")
        fig.colorbar(image,ax=axes[:,2].tolist(),fraction=.025,pad=.02)
        # Explicit layout avoids colorbar overlap with tight_layout.
        fig.subplots_adjust(left=.06,right=.88,bottom=.04,top=.93,hspace=.4,wspace=.3)
        fig.savefig(F/('diagnostic_score_'+kind+'.png'),dpi=150);fig.savefig(F/('diagnostic_score_'+kind+'.svg'),metadata={'Date':None});plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for kind in ['linear','cosine']:
        rows=diag[kind]['trajectory_modes'];axes[0].plot([r['fraction'] for r in rows],[r['agreement_all'] for r in rows],'o-',label=kind)
        saved=torch.load(Path('runs/lab7')/diag[kind]['trajectory_RUN_ID']/'trajectories1000.pt',weights_only=False)
        path=torch.stack([saved[t] for t in sorted(saved,reverse=True)]).numpy()
        for j in range(20):axes[1].plot(path[:,j,0],path[:,j,1],alpha=.15,lw=.6)
    axes[0].set(xlabel='t/T (denoising proceeds right to left)',ylabel='Agreement with final geometric mode',ylim=(0,1));axes[0].legend()
    axes[1].set(title='First20 fixed DDPM paths per schedule',xlabel='standardized x',ylabel='standardized y');save(fig,'diagnostic_trajectories')
    if e3 is None or sup is None:
        print('Exploratory plots generated; E3/supplementary plots await their raw results',flush=True)
        return
    table('E3_models.csv',[{'seed':p['seed'],'arm':a,**p[a]} for p in e3['pairs'] for a in ['A','B']])
    table('E3_pairs.csv',[{'seed':p['seed'],'A_Dmode':p['A']['Dmode'],'B_Dmode':p['B']['Dmode'],'delta':delta,
         'A_RUN_ID':p['A']['RUN_ID'],'B_RUN_ID':p['B']['RUN_ID']} for p,delta in zip(e3['pairs'],e3['deltas'])])
    ref=np.asarray(noise['reference_frequencies']);x=np.arange(len(ref));fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for a,p in zip(axes,e3['pairs']):
        a.bar(x-.25,ref,.25,label='100k target');a.bar(x,p['A']['mode_frequencies'],.25,label='A');a.bar(x+.25,p['B']['mode_frequencies'],.25,label='B')
        a.set(title=f"seed{p['seed']}: A {e3['A']['sampler']}, B {e3['B']['sampler']} K={e3['B']['K']}",xlabel='Mode (0 rare; last outlier)',ylabel='Frequency');a.set_xticks(x)
    axes[0].legend(fontsize=8);save(fig,'E3_mode_frequencies')
    fig,a=plt.subplots(figsize=(7,3.6));a.bar([1,2,3],e3['deltas']);a.axhline(0,c='black',lw=.8)
    a.errorbar([4],[e3['mean']],yerr=[e3['sample_sd']],fmt='o',capsize=5,c='black',label='mean +/- sample SD')
    a.set_xticks([1,2,3,4],['seed1','seed2','seed3','mean']);a.set(ylabel='Dmode(A) - Dmode(B)',title=e3['teacher_conclusion']);a.legend();save(fig,'E3_paired_difference')
    table('supplement_rare.csv',[{k:v for k,v in r.items() if k!='quality'}|r['quality'] for r in sup['rare_boundaries']])
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));r=sup['rare_boundaries'];axes[0].plot([v['pi0'] for v in r],[v['quality']['rare_frequency'] for v in r],'o-',label='Generated rare frequency')
    axes[0].plot([v['pi0'] for v in r],[v['pi0']/2 for v in r],'--',label='pi0/2 shorthand');axes[0].set(xlabel='Target pi0',ylabel='Mode0 frequency');axes[0].legend(fontsize=8)
    axes[1].bar([str(v['pi0']) for v in r],[v['train_component0_count'] for v in r]);axes[1].set(xlabel='Target pi0',ylabel='Actual rare component train count');save(fig,'supplement_rare')
    if (R/'counterfactual_answers.json').exists():
        answers=read(R/'counterfactual_answers.json')
        controls=[('linear DDIM K=5 eta=0',answers[3]['DDIM5']['linear']),
                  ('cosine DDIM K=5 eta=0',answers[3]['DDIM5']['cosine']),
                  ('linear DDPM T=10000 eta=n/a',sup['Q1']['quality']),
                  ('zero epsilon DDPM T=1000 eta=n/a',sup['Q2']['quality'])]
        fig,axes=plt.subplots(2,2,figsize=(9,8))
        for ax,(title,r) in zip(axes.flat,controls):
            x=pts(r['RUN_ID']);outside=np.mean(np.any(np.abs(x)>2,axis=1))
            ax.scatter(data[:,0],data[:,1],s=.7,c='gray',alpha=.15,rasterized=True)
            ax.scatter(x[:,0],x[:,1],s=1,alpha=.3,rasterized=True)
            ax.set(title=title+f"\noutside view={100*outside:.1f}%; Dmode={r['Dmode']:.3f}",xlim=(-2,2),ylim=(-2,2),aspect='equal')
        save(fig,'counterfactual_controls')
    provenance={'source_records':{str(p):file_hash(p) for p in [R/'E1.json',R/'E2.json',R/'E3.json',R/'diagnostics.json',R/'supplements.json']},
        'figure_hashes':{p.name:file_hash(p) for p in sorted(F.glob('*')) if p.suffix in ['.png','.svg']},
        'figure_groups':len(list(F.glob('*.png'))),'note':'generated from raw saved records; no optimization or sampling'}
    write_json(R/'figure_provenance.json',provenance);print('Generated figures:',provenance['figure_groups'],flush=True)


if __name__=='__main__':main()
