"""Provenance/evaluation helpers. Optimization exclusively uses existing Runner."""
import copy,json,math,time
from pathlib import Path
import torch,yaml
from src.config import validate
from src.data.generative import datasets
from src.models.small_cnn import build_model
from src.engine.runner import ExperimentRunner
from src.utils.io import write_json,file_hash,digest
from src.utils.provenance import git_state,create_run
from src.generative.distributions import distribution,make_roles
from src.generative.samplers import langevin_sample
from src.generative.density import density_grid
from src.diagnostics.density_metrics import bandwidth,kernel_mean,sample_metrics
from src.diagnostics.mcmc import mode_switches,quadratic_chains
from src.diagnostics.invertibility import inverse_error,autograd_logdet
from src.analysis.generative_comparison import nearest_centers

ROOT=Path('experiments/lab6')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def config(kind='ebm',target='two_blobs',seed=1,role='E1',width=None,depth=2,eta=None,purpose='formal'):
    c=yaml.safe_load(Path('configs/lab6.yaml').read_text(encoding='utf-8'))
    c['experiment'].update(name=role,role=role,seed=seed,purpose=purpose);c['data']['target']=target
    c['model'].update(name='energy_model' if kind=='ebm' else 'normalizing_flow',width=width or (128 if kind=='ebm' else 64),depth=3 if kind=='ebm' else depth)
    if eta is not None:c['sampling']['eta']=eta
    validate(c);return c

def train(c):validate(c);return str(ExperimentRunner(c).run()).replace('\\','/')

def operation(c,model,role,parent=None):
    torch.set_num_threads(c['runtime']['threads'])
    code=git_state()
    if c['experiment']['purpose']=='formal' and code['dirty']:raise RuntimeError('Formal operation requires committed clean source/config')
    _,info=datasets(c);path,m=create_run(c,code,info,model,parent)
    m.update(status='COMPLETED_'+role,global_step=0,completed_epoch=0);write_json(path/'manifest.json',m)
    return path,m

def source_model(source,dtype=None):
    p=Path(source);c=yaml.safe_load((p/'config.resolved.yaml').read_text(encoding='utf-8'))
    model=build_model(c['model']);state=torch.load(p/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);model=model.to(c['runtime']['device']).eval()
    if dtype:model=model.to(dtype=dtype)
    return model,c,state

def reference_protocol(target_name):
    torch.set_num_threads(2)
    p=ROOT/'metrics_protocol'/f'{target_name}.json'
    if p.exists():return read(p)
    target=distribution(target_name);roles=make_roles(target_name);h,indices=bandwidth(roles['train'])
    ref=roles['test'];self_mean=kernel_mean(ref,ref,h)
    c=config(target=target_name,role='metric_protocol');model=build_model(c['model']);path,_=operation(c,model,'METRIC_PROTOCOL')
    result={'RUN_ID':path.name,'target':target_name,'bandwidth':h,'bandwidth_seed':604,'train_indices':indices.tolist(),
        'bandwidth_method':'median Euclidean distance, all1999000distinctpairs of2000train points','fixed_test_self_kernel_mean':self_mean,
        'test_count':5000,'true_mode_probabilities_outlier_first':target.mode_probabilities().tolist()}
    write_json(path/'summary.json',result);write_json(p,result);return result

def noise_baseline(target_name):
    p=ROOT/'noise'/f'{target_name}.json'
    if p.exists():return read(p)
    protocol=reference_protocol(target_name);roles=make_roles(target_name);target=distribution(target_name);rows=[]
    for i in range(5):
        seed=6101+i;fresh=distribution(target_name,seed).sample(5000)
        c=config(target=target_name,seed=seed,role='metric_noise');model=build_model(c['model']);path,_=operation(c,model,'ANALYTIC_NOISE_BASELINE')
        metrics=sample_metrics(fresh,roles['test'],target,protocol['bandwidth'],protocol['fixed_test_self_kernel_mean'])
        row={'RUN_ID':path.name,'seed':seed,'sample_count':5000,**metrics}
        torch.save(fresh,path/'fresh_target_samples.pt');write_json(path/'summary.json',row);rows.append(row)
    result={'target':target_name,'repetitions':rows,'noise_MMD2':sum(r['MMD2'] for r in rows)/5,
        'noise_Dmode':sum(r['Dmode'] for r in rows)/5,'protocol_RUN_ID':protocol['RUN_ID'],'noise_definition':'mean of five fresh5000target vs fixed5000test comparisons'}
    write_json(p,result);return result

def evaluate(source,eta=None,seed=7001,center_probe=False):
    torch.set_num_threads(2)
    model,c,state=source_model(source);kind='ebm' if c['model']['name']=='energy_model' else 'flow'
    c=copy.deepcopy(c);c['experiment']['role']='density_evaluation'
    if eta is not None:c['sampling']['eta']=eta
    path,_=operation(c,model,'DENSITY_EVALUATION',Path(source).name)
    target=distribution(c['data']['target']);roles=make_roles(target.name);protocol=reference_protocol(target.name)
    p=next(model.parameters());gen=torch.Generator().manual_seed(seed)
    if p.device.type=='cuda':torch.cuda.synchronize()
    start=time.perf_counter();sampler_info={}
    if kind=='ebm':
        x0=torch.randn(5000,2,generator=gen).to(p)
        samples,traj,sampler_info=langevin_sample(model,x0,c['sampling']['eta'],1.,50,target.domain,gen,True)
    else:
        with torch.no_grad():samples=model.sample(5000,gen)
    if p.device.type=='cuda':torch.cuda.synchronize()
    sample_time=time.perf_counter()-start;samples=samples.detach().cpu()
    metrics=sample_metrics(samples,roles['test'],target,protocol['bandwidth'],protocol['fixed_test_self_kernel_mean'])
    grid_info,grid_data=density_grid(model,target,kind)
    result={'RUN_ID':path.name,'source_RUN_ID':Path(source).name,'target':target.name,'kind':kind,'seed':c['experiment']['seed'],
        'checkpoint_sha256':file_hash(Path(source)/'checkpoints/last.pt'),'source_global_step':state['global_step'],
        'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':read(Path(source)/'summary.json')['training_seconds'],
        'sample_count':5000,'sample_seed':seed,'sample_seconds':sample_time,'metric_protocol_RUN_ID':protocol['RUN_ID'],
        **metrics,**grid_info,'sampler':sampler_info}
    test=roles['test'].to(p);val=roles['val'].to(p)
    with torch.no_grad():
        if kind=='flow':
            vnll=-model.log_prob(val).cpu().double();tnll=-model.log_prob(test).cpu().double()
            result.update(val_NLL=float(vnll.mean()),val_NLL_SEM=float(vnll.std(unbiased=True)/math.sqrt(len(vnll))),
                exact_test_NLL=float(tnll.mean()),test_NLL_SEM=float(tnll.std(unbiased=True)/math.sqrt(len(tnll))))
            torch.save(vnll,path/'val_pointwise_nll.pt');torch.save(tnll,path/'test_pointwise_nll.pt')
        else:
            result['mean_test_energy']=float(model(test).mean());result['center_energies']=model(target.centers.to(p)).cpu().tolist()
            result['origin_energy']=float(model(torch.zeros(1,2).to(p)))
            result['sampler'].update(mode_switches(traj,target));torch.save(traj.detach().cpu(),path/'sampling_trajectory.pt')
            low=torch.tensor([-15.,-15.]);high=-low
            result['outside_final_fraction']=float(((samples<low)|(samples>high)).any(-1).float().mean())
            result['stable']=sampler_info['finite'] and result['outside_final_fraction']<=.05
    if center_probe:
        g=torch.Generator().manual_seed(7101);x0=target.centers[0].float().repeat(5000,1).to(p)
        end,ctr,info=langevin_sample(model,x0,c['sampling']['eta'],1.,50,target.domain,g,True)
        result['center_start_mixing']={**mode_switches(ctr,target),'chains':5000,'start_center':0,'seed':7101}
        torch.save(ctr.detach().cpu(),path/'center_start_trajectory.pt')
    # Float64 diagnostics use a parameter-identical copy of learned float32 weights.
    if kind=='flow':
        double=copy.deepcopy(model).cpu().double();x=roles['test'].double()
        with torch.no_grad():err=inverse_error(double,x)
        jac=autograd_logdet(double,x[:100]);g=torch.Generator().manual_seed(6201)
        angles=torch.rand(100,generator=g,dtype=torch.double)*2*math.pi
        far=20*torch.stack([angles.cos(),angles.sin()],-1)
        with torch.no_grad():far_err=inverse_error(double,far)
        result['invertibility']={'dtype':'float64 with converted learned weights','test_points':5000,'median_epsilon':float(err.median()),
            'max_epsilon':float(err.max()),'autograd_points':100,'max_logdet_error':jac['max_error'],
            'norm20_points':100,'norm20_seed':6201,'norm20_median_epsilon':float(far_err.median()),'norm20_max_epsilon':float(far_err.max())}
        torch.save({'epsilon':err,'far_points':far,'far_epsilon':far_err,'jacobian':jac},path/'invertibility.pt')
    c1,c2,mid,ids=nearest_centers(target);line=c1[None]+torch.linspace(0,1,201,dtype=torch.double)[:,None]*(c2-c1)[None]
    with torch.no_grad():
        mid_input=mid[None].to(p);model_log=float(model.log_prob(mid_input)) if kind=='flow' else float(-model(mid_input)-grid_info['log_Z_hat_omega'])
        line_log=model.log_prob(line.to(p)).cpu().double() if kind=='flow' else -model(line.to(p)).cpu().double()-grid_info['log_Z_hat_omega']
    result['midpoint']={'center_indices':ids,'centers':[c1.tolist(),c2.tolist()],'point':mid.tolist(),'true_log_density':float(target.log_prob(mid)),
        'model_log_density':model_log,'model_density_kind':grid_info['density_kind'],'log_density_excess_over_true':model_log-float(target.log_prob(mid))}
    torch.save({'points':line,'true_log_density':target.log_prob(line),'model_log_density':line_log},path/'bridge_line.pt')
    torch.save(samples,path/'generated_samples.pt');torch.save(grid_data,path/'density_grid.pt')
    write_json(path/'summary.json',result);return result

def quadratic_evaluation(eta,seed,control_reference_eta=None):
    c=config(seed=seed,role='quadratic_control' if control_reference_eta else 'quadratic_variance')
    c['quadratic_protocol']={'eta':eta,'control_reference_eta':control_reference_eta,'chains':8,'burn_in':10000,'draws':200000,'boundary':None}
    model=build_model(c['model']);path,_=operation(c,model,'QUADRATIC_FINITE_CHAIN')
    result,traj,noise=quadratic_chains(eta,seed=seed,control_reference_eta=control_reference_eta)
    result.update(RUN_ID=path.name,optimizer_steps=0)
    torch.save({'trajectory':traj,'standard_normal_innovations':noise},path/'quadratic_chains.pt')
    write_json(path/'summary.json',result);return result
