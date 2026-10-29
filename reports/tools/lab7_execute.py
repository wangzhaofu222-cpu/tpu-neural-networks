"""Persistent serial Lab7 experiment controller; all optimization uses ExperimentRunner."""
import argparse,copy,math
from pathlib import Path
import torch
from src.diffusion.schedules import make_schedule,terminal_kl
from src.diffusion.forward import q_sample
from src.data.gmm2d import standardize,target,ExactPredictor
from src.utils.io import write_json
from reports.tools.lab7_common import R,base,read,registry,update,train,evaluate,observation,metric_protocol,source_model


def e1():
    if (R/'E1.json').exists():return
    c=base();protocol,roles,stats=metric_protocol(c);z=standardize(roles['train'],stats)
    diagnostic=observation('E1_schedules_forward',c,'schedule_theory_and_forward_moments')
    schedules={};clouds={}
    for kind in ['linear','cosine']:
        cfg={**c['diffusion'],'kind':kind};s=make_schedule(cfg)
        rows=[];g=torch.Generator().manual_seed(7301)
        for frac in [.1,.25,.5,.75,1.]:
            t=round(frac*s.T);noise=torch.randn(10000,2,generator=g,dtype=torch.float64)
            xt,_=q_sample(z[:1].expand(10000,2),t,s,noise);v=1-s.alpha_bar[t];expected=s.alpha_bar[t].sqrt()*z[0]
            mean_error=(xt.mean(0)-expected).abs();var_error=(xt.var(0)/v-1).abs()
            rows.append({'t':t,'fraction':frac,'mean_measured':xt.mean(0).tolist(),'mean_expected':expected.tolist(),
                'variance_measured':xt.var(0).tolist(),'variance_expected':float(v),
                'mean_error':mean_error.tolist(),'mean_threshold':float(4*(v/10000).sqrt()),
                'variance_relative_error':var_error.tolist(),'variance_relative_threshold':4*math.sqrt(2/10000),
                'PASS':bool((mean_error<=4*(v/10000).sqrt()).all() and (var_error<=4*math.sqrt(2/10000)).all())})
            cloud,_=q_sample(z,t,s,torch.randn(z.shape,generator=g,dtype=torch.float64));clouds[kind+'_'+str(t)]=cloud
        torch.save({k:getattr(s,k) for k in ['beta','alpha','alpha_bar','snr','log_snr','posterior_var']},diagnostic/(kind+'_schedule.pt'))
        schedules[kind]={'alpha_bar_T':float(s.alpha_bar[-1]),'train_mean_terminal_KL':terminal_kl(s,z),
            'SNR_less_point01_fraction':float((s.snr[1:]<.01).double().mean()),'forward_moments':rows,
            'all_forward_moments_PASS':all(r['PASS'] for r in rows),'RUN_ID':diagnostic.name}
    torch.save(clouds,diagnostic/'forward_clouds_full_train.pt');write_json(diagnostic/'summary.json',schedules)
    results={}
    for kind in ['linear','cosine']:
        cfg=copy.deepcopy(c);cfg['diffusion']['kind']=kind;cfg['diffusion']['alpha_bar_T']=schedules[kind]['alpha_bar_T']
        source=train('E1_'+kind,cfg)
        result=evaluate('E1_'+kind+'_DDPM',source)
        results[kind]={'training_source':str(source),'training_summary':read(source/'summary.json'),'quality':result}
    cfg=copy.deepcopy(c['diffusion']);cfg['T']=cfg['T']//10
    short=make_schedule(cfg,diagnostic=True)
    boundary=evaluate('E1_short_T_exact_DDPM',results['linear']['training_source'],predictor='exact',schedule_override=cfg)
    boundary['terminal_KL']=terminal_kl(short,z)
    full=evaluate('E1_full_T_exact_DDPM',results['linear']['training_source'],predictor='exact')
    write_json(R/'E1.json',{'theory':schedules,'models':results,'boundary_short':boundary,'boundary_full':full,
        'forward_seed':7301,'fixed_x0_standardized':z[0].tolist()})
    print('E1 COMPLETE',flush=True)


def e2():
    if (R/'E2.json').exists():return
    source=registry()['training']['E1_linear'];T=base()['diffusion']['T'];rows=[]
    for predictor in ['network','exact']:
        for sampler,K in [('ddpm',T)]+[('ddim',k) for k in [5,10,20,50,100,T]]:
            key=f'E2_{predictor}_{sampler}_{K}'
            rows.append(evaluate(key,source,sampler,K,predictor,timing_repeats=5,warmup=True))
    baseline=next(r for r in rows if r['predictor']=='network' and r['sampler']=='ddim' and r['K']==T)
    extra=[evaluate('E2_network_DDIM_T_initial_'+str(s),source,'ddim',T,initial_seed=s) for s in [7402,7403]]
    v=torch.tensor([baseline['MMD2']]+[r['MMD2'] for r in extra],dtype=torch.float64);sd=float(v.std())
    candidates=[r for r in rows if r['predictor']=='network' and r['sampler']=='ddim' and abs(r['MMD2']-baseline['MMD2'])<=2*sd]
    low=[evaluate(f'E2_low_{predictor}_{K}',source,'ddim',K,predictor) for predictor in ['network','exact'] for K in [1,2]]
    protocol,_,_=metric_protocol(base());threshold=protocol['MMD2_noise_mean']+3*protocol['MMD2_noise_sd']
    timing={}
    for pred in ['network','exact']:
        group=[r for r in rows if r['predictor']==pred and r['K']>=10]
        per=torch.tensor([r['seconds_median']/r['NFE'] for r in group],dtype=torch.float64);median=float(per.median())
        timing[pred]={'median_seconds_per_NFE':median,'conditions':[{'RUN_ID':r['RUN_ID'],'relative_deviation':abs(r['seconds_median']/r['NFE']/median-1),
            'PASS':abs(r['seconds_median']/r['NFE']/median-1)<=.2} for r in group]}
    write_json(R/'E2.json',{'rows':rows,'additional_initial_noises':extra,'MMD_T_values':v.tolist(),'s_MMD':sd,
        'smallest_effective_K':min(r['K'] for r in candidates),'selection_threshold':2*sd,'low_K':low,
        'exact_noise_threshold':threshold,'exact_full_acceptance':[{'RUN_ID':r['RUN_ID'],'PASS':r['MMD2']<=threshold}
            for r in rows if r['predictor']=='exact' and r['K']==T],'timing_acceptance':timing})
    print('E2 COMPLETE',flush=True)


def diagnostics():
    if (R/'diagnostics.json').exists():return
    from src.diagnostics.timestep_error import timestep_errors
    from src.diagnostics.score_field import score_field
    from src.diagnostics.trajectory import trajectory_modes
    out={};T=base()['diffusion']['T']
    for kind in ['linear','cosine']:
        source=registry()['training']['E1_'+kind];model,cfg,_=source_model(source)
        s=make_schedule(cfg['diffusion']).to(cfg['runtime']['device']);_,roles,stats=metric_protocol(cfg)
        val=standardize(roles['val'],stats).float().to(cfg['runtime']['device'])
        noise=torch.randn(val.shape,generator=torch.Generator().manual_seed(7701)).to(val.device)
        exact=ExactPredictor(target(cfg),stats,s)
        p=observation('diagnostics_'+kind,cfg,'timestep_score_field_trajectory')
        times=sorted(set(torch.linspace(1,T,25).round().long().tolist()))
        errors=timestep_errors(model,exact,s,val,noise,times)
        exact_errors=timestep_errors(exact,exact,s,val,noise,times)
        axis=torch.linspace(-3.5,3.5,41);xx,yy=torch.meshgrid(axis,axis,indexing='ij')
        points=torch.stack([xx.flatten(),yy.flatten()],1).to(val.device)
        fields=[score_field(model,exact,s,points,round(f*T)) for f in [.01,.1,.4,.8]]
        torch.save(fields,p/'score_fields.pt');torch.save({'val':val.cpu(),'noise':noise.cpu()},p/'diagnostic_inputs.pt')
        write_json(p/'timestep_errors.json',{'network':errors,'exact':exact_errors})
        result=evaluate('diagnostic_trajectory_'+kind,source,save_full_1000=True)
        saved=torch.load(Path('runs/lab7')/result['RUN_ID']/'trajectories1000.pt',weights_only=False)
        modes=trajectory_modes(saved,s,target(cfg),stats)
        summaries=[{k:v for k,v in f.items() if not isinstance(v,torch.Tensor)} for f in fields]
        out[kind]={'RUN_ID':p.name,'trajectory_RUN_ID':result['RUN_ID'],'errors':errors,'exact_errors':exact_errors,
            'fields':summaries,'trajectory_modes':modes}
        write_json(p/'summary.json',out[kind])
    write_json(R/'diagnostics.json',out);print('DIAGNOSTICS COMPLETE',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['e1','e2','diagnostics']);args=parser.parse_args()
    {'e1':e1,'e2':e2,'diagnostics':diagnostics}[args.stage]()
