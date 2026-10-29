import copy,json,time
from pathlib import Path
import torch,yaml
from src.data.gmm2d import target,roles_and_stats,ExactPredictor,inverse_standardize
from src.diffusion.schedules import make_schedule
from src.diffusion.samplers import ddpm_sample,ddim_sample
from src.models.small_cnn import build_model
from src.diagnostics.sample_quality import quality,frequencies
from src.diagnostics.density_metrics import kernel_mean
from src.utils.io import write_json,digest,file_hash
from src.utils.provenance import create_run,git_state

R=Path('experiments/lab7')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def base():return yaml.safe_load(Path('configs/lab7.yaml').read_text(encoding='utf-8'))


def registry():
    return read(R/'run_index.json') if (R/'run_index.json').exists() else {'training':{},'evaluations':{},'observations':{}}


def update(group,key,value):
    idx=registry();idx[group][key]=value;write_json(R/'run_index.json',idx)


def source_model(source):
    source=Path(source);cfg=yaml.safe_load((source/'config.resolved.yaml').read_text(encoding='utf-8'))
    model=build_model(cfg['model']).to(cfg['runtime']['device'])
    state=torch.load(source/'checkpoints/last.pt',map_location=cfg['runtime']['device'],weights_only=False)
    model.load_state_dict(state['model']);model.eval()
    return model,cfg,state


def train(key,cfg):
    from src.engine.runner import ExperimentRunner
    idx=registry()
    if key in idx['training']:
        p=Path(idx['training'][key]);assert read(p/'manifest.json')['global_step']==20000;return p
    cfg=copy.deepcopy(cfg);cfg['experiment'].update(purpose='formal',role=key)
    # Recover completed orphan runs without retraining if a controller stopped before indexing.
    for p in Path('runs/lab7').glob('*'):
        if (p/'manifest.json').exists():
            m=read(p/'manifest.json')
            if m['config_hash']==digest(cfg) and m['status']=='COMPLETED' and m.get('global_step')==20000:
                update('training',key,str(p));return p
    p=ExperimentRunner(cfg).run();assert read(p/'manifest.json')['global_step']==20000
    update('training',key,str(p));return p


def metric_protocol(cfg):
    root=R/'metrics';root.mkdir(exist_ok=True)
    key=str(cfg['data']['pi0']);path=root/(key+'.json')
    roles,stats=roles_and_stats(cfg);dist=target(cfg)
    main=root/'bandwidth.json'
    if main.exists():h=read(main)['h']
    else:
        raw=roles_and_stats(base())[0]['train']
        distances=torch.pdist(raw.double());h=float(distances.median());del distances
        write_json(main,{'h':h,'source':'all20000rawtrain distinct pair distances,exact torch.pdist median',
                         'pairs':199990000,'train_seed':701,'train_config':base()['data'],'MMD_estimator':'V including diagonal'})
    if path.exists():return read(path),roles,stats
    reference=dist.sample(100000,704);ref=frequencies(reference,dist)
    data=root/key;data.mkdir(exist_ok=True)
    torch.save(reference,data/'reference100000.pt')
    selfkernel=kernel_mean(roles['test'],roles['test'],h)
    values=[]
    for seed in range(7601,7606):
        fresh=dist.sample(5000,seed);q=quality(fresh,roles['test'],dist,ref,h,selfkernel)
        cfgobs=copy.deepcopy(cfg);cfgobs['experiment'].update(purpose='evaluation',role='noise_'+str(seed));cfgobs['evaluation']={'kind':'fresh_target_metric_noise','seed':seed}
        dummy=build_model(cfg['model']).to(cfg['runtime']['device'])
        p,m=create_run(cfgobs,git_state(),{'data_fingerprint':digest(cfg['data']),'split_fingerprint':digest(cfg['data']['seeds'])},dummy)
        torch.save(fresh,p/'generated_samples.pt');write_json(p/'summary.json',q)
        m.update(status='COMPLETED_EVALUATION',parameters=0,trainable_parameters=0,optimizer_steps=0);write_json(p/'manifest.json',m)
        values.append({'seed':seed,'RUN_ID':p.name,**q});update('observations','noise_'+key+'_'+str(seed),str(p))
    result={'h':h,'pi0':dist.pi0,'reference_seed':704,'reference_n':100000,'reference_frequencies':ref.tolist(),
            'test_self_kernel':selfkernel,'noise':values,'reference_sha256':file_hash(data/'reference100000.pt')}
    for metric in ['MMD2','Dmode']:
        v=torch.tensor([r[metric] for r in values],dtype=torch.float64)
        result[metric+'_noise_mean']=float(v.mean());result[metric+'_noise_sd']=float(v.std())
    write_json(path,result);return result,roles,stats


def observation(key,cfg,kind):
    c=copy.deepcopy(cfg);c['experiment'].update(purpose='evaluation',role=key);c['evaluation']={'kind':kind}
    dummy=build_model(c['model']).to(c['runtime']['device'])
    p,m=create_run(c,git_state(),{'data_fingerprint':digest(c['data']),'split_fingerprint':digest(c['data']['seeds'])},dummy)
    m.update(status='COMPLETED_DIAGNOSTIC',parameters=0,trainable_parameters=0,optimizer_steps=0)
    write_json(p/'manifest.json',m);update('observations',key,str(p));return p


def evaluate(key,source,sampler='ddpm',K=None,predictor='network',initial_seed=7401,noise_seed=7501,
             schedule_override=None,timing_repeats=1,warmup=False,keep=None,save_full_1000=False):
    idx=registry()
    if key in idx['evaluations']:return idx['evaluations'][key]
    model,cfg,state=source_model(source)
    if schedule_override:cfg['diffusion']=copy.deepcopy(schedule_override)
    short=cfg['diffusion']['T']<cfg['profile']['lab7_T_beta'][0]
    s=make_schedule(cfg['diffusion'],diagnostic=short).to(cfg['runtime']['device'])
    protocol,roles,stats=metric_protocol(cfg);dist=target(cfg)
    device=torch.device(cfg['runtime']['device']);dtype=torch.float64 if predictor=='exact' else torch.float32
    xT=torch.randn(5000,2,generator=torch.Generator().manual_seed(initial_seed),dtype=torch.float64).to(device=device,dtype=dtype)
    fn=model if predictor=='network' else ExactPredictor(dist,stats,s) if predictor=='exact' else lambda x,t:torch.zeros_like(x)
    K=s.T if sampler=='ddpm' else int(K)
    tau=list(range(1,s.T+1)) if sampler=='ddpm' else __import__('src.diffusion.samplers',fromlist=['make_tau']).make_tau(s.T,K)
    evalcfg=copy.deepcopy(cfg);evalcfg['experiment'].update(purpose='evaluation',role=key)
    evalcfg['evaluation']={'predictor':predictor,'source_RUN_ID':Path(source).name,'sampler':sampler,'K':K,'eta':0. if sampler=='ddim' else None,
        'tau':tau,'initial_seed':initial_seed,'noise_seed':noise_seed,'samples':5000,'timing_repeats':timing_repeats,'warmup':warmup,
        'terminal_protocol_valid':s.protocol_valid,'alpha_bar_T':float(s.alpha_bar[-1]),'variance':'posterior','clipping':False}
    p,m=create_run(evalcfg,git_state(),{'data_fingerprint':digest(cfg['data']),'split_fingerprint':digest(cfg['data']['seeds']),'normalization':stats},model)
    def sample(x,save=None):
        g=torch.Generator(device=device).manual_seed(noise_seed)
        if sampler=='ddpm':return ddpm_sample(fn,s,x,keep=save,generator=g)
        return ddim_sample(fn,s,x,K,eta=0.,keep=save,generator=g)
    if warmup:sample(xT)
    times=[];first=None;max_difference=0.
    for repeat in range(timing_repeats):
        output,info=sample(xT,keep if repeat==0 else None)
        times.append(info['seconds'])
        if first is None:first=output.detach().cpu();saved=info['saved']
        else:max_difference=max(max_difference,float((first-output.detach().cpu()).abs().max()))
    raw=inverse_standardize(first.double(),stats)
    q=quality(raw,roles['test'],dist,protocol['reference_frequencies'],protocol['h'],protocol['test_self_kernel'])
    result={'RUN_ID':p.name,'source_RUN_ID':Path(source).name,'source_checkpoint_sha256':file_hash(Path(source)/'checkpoints/last.pt'),
            **evalcfg['evaluation'],**q,'seconds_repetitions':times,'seconds_median':float(torch.tensor(times,dtype=torch.float64).median()),
            'NFE':info['NFE'],'total_predictor_calls_per_pass':info['total_predictor_calls'],'repeat_max_difference':max_difference,
            'device':str(device),'sampling_dtype':str(dtype)}
    torch.save(first,p/'standardized_samples.pt');torch.save(raw,p/'generated_samples.pt');torch.save(xT.cpu(),p/'initial_noise.pt')
    if keep:torch.save(saved,p/'saved_states.pt')
    if save_full_1000:
        full,trajectory=sample(xT[:1000],list(range(s.T+1)));torch.save(trajectory['saved'],p/'trajectories1000.pt')
    write_json(p/'summary.json',result);m.update(status='COMPLETED_EVALUATION',optimizer_steps=0,source_RUN_ID=Path(source).name)
    if predictor!='network':m.update(parameters=0,trainable_parameters=0)
    write_json(p/'manifest.json',m);update('evaluations',key,result)
    print('EVALUATED',key,'MMD2',q['MMD2'],'Dmode',q['Dmode'],'Cov',q['Cov'],flush=True)
    return result
