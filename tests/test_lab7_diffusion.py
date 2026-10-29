import copy
import json
from pathlib import Path
import pytest
import torch
import yaml
from src.diffusion.schedules import make_schedule
from src.diffusion.forward import q_sample
from src.diffusion.parameterizations import EpsilonParameterization
from src.diffusion.samplers import ddpm_sample,ddim_sample,make_tau
from src.data.gmm2d import RareGMM,ExactPredictor,roles_and_stats,standardize,inverse_standardize
from src.models.diffusion.mlp_eps import EpsilonMLP


def schedule(kind='linear',T=1000):
    return make_schedule({'kind':kind,'T':T,'beta_min':.0001,'beta_max':.02})


@pytest.mark.parametrize('kind',['linear','cosine'])
def test_schedule_group1(kind):
    s=schedule(kind)
    assert s.beta.dtype==torch.float64 and len(s.beta)==1001
    assert ((s.beta[1:]>0)&(s.beta[1:]<1)).all()
    assert (s.alpha_bar[1:]<s.alpha_bar[:-1]).all()
    assert (s.log_snr[2:]<s.log_snr[1:-1]).all()
    assert s.alpha_bar[0]==1 and s.posterior_var[1]==0 and s.alpha_bar[-1]<=1e-3
    assert torch.equal(s.alpha_bar,torch.cumprod(1-s.beta,0))
    assert torch.isinf(s.log_snr[0])


def test_short_terminal_gate():
    cfg={'kind':'linear','T':100,'beta_min':.0001,'beta_max':.02}
    with pytest.raises(ValueError):make_schedule(cfg)
    assert not make_schedule(cfg,diagnostic=True).protocol_valid


@pytest.mark.parametrize('shape',[(11,2),(11,3,4,4)])
def test_q_broadcast(shape):
    s=schedule();g=torch.Generator().manual_seed(711)
    x=torch.randn(shape,generator=g,dtype=torch.float64);n=torch.randn(shape,generator=g,dtype=torch.float64)
    t=torch.arange(1,12)*80;a=s.alpha_bar[t].reshape((-1,)+(1,)*(len(shape)-1))
    xt,used=q_sample(x,t,s,n)
    assert torch.equal(used,n) and torch.equal(xt,a.sqrt()*x+(1-a).sqrt()*n)


def test_q_moments_group2():
    s=schedule();g=torch.Generator().manual_seed(713)
    for t in [100,250,500,750,1000]:
        x=torch.tensor([.2,-.7],dtype=torch.float64).expand(10000,2)
        n=torch.randn(x.shape,generator=g,dtype=x.dtype);xt,_=q_sample(x,t,s,n)
        v=1-s.alpha_bar[t]
        assert ((xt.mean(0)-s.alpha_bar[t].sqrt()*x[0]).abs()<=4*(v/10000).sqrt()).all()
        assert ((xt.var(0)/v-1).abs()<=4*(torch.tensor(2/10000).sqrt())).all()


@pytest.mark.parametrize('kind',['linear','cosine'])
def test_reconstruction_identity_all_times(kind):
    s=schedule(kind);p=EpsilonParameterization(s);g=torch.Generator().manual_seed(714)
    x=torch.randn(1000,2,generator=g,dtype=torch.float64);n=torch.randn(1000,2,generator=g,dtype=torch.float64)
    t=torch.arange(1,1001);xt,_=q_sample(x,t,s,n)
    assert (p.to_x0(n,xt,t)-x).abs().max()<1e-8
    pred=n+.03
    lhs=(x-p.to_x0(pred,xt,t)).square().mean(1)
    rhs=(n-pred).square().mean(1)/s.snr[t]
    assert torch.allclose(lhs,rhs,rtol=1e-7,atol=1e-10)


@pytest.mark.parametrize('t',[0,1,100,1000])
def test_exact_score_finite_difference(t):
    d=RareGMM(6,.1,.02);s=schedule();stats={'mean':[.12,-.03],'scale':[.6,.8]}
    e=ExactPredictor(d,stats,s);g=torch.Generator().manual_seed(715)
    x=torch.randn(32,2,generator=g,dtype=torch.float64)
    score=e.score(x,t);fd=[];h=1e-5
    for j in range(2):
        shift=torch.zeros_like(x);shift[:,j]=h
        fd.append((e.log_prob(x+shift,t)-e.log_prob(x-shift,t))/(2*h))
    fd=torch.stack(fd,1)
    assert (score-fd).norm()/score.norm()<1e-4
    with torch.no_grad():assert torch.isfinite(e(x,t)).all()
    assert torch.isfinite(e.log_prob(torch.full_like(x,1e4),t)).all()


def test_sampler_group4_equivalence_and_immutability():
    s=schedule();g=torch.Generator().manual_seed(716)
    model=EpsilonMLP({'width':8,'depth':2,'time_dim':8}).double().eval()
    original=copy.deepcopy(model.state_dict());x=torch.randn(9,2,generator=g,dtype=torch.float64)
    z=torch.randn(1001,9,2,generator=g,dtype=torch.float64)
    a,ia=ddpm_sample(model,s,x,z=z,keep=[0,1000]);b,ib=ddim_sample(model,s,x,1000,eta=1,z=z)
    assert torch.allclose(a,b,rtol=0,atol=1e-6)
    assert ia['NFE']==ib['NFE']==1000 and not a.requires_grad
    c,_=ddim_sample(model,s,x,10);d,_=ddim_sample(model,s,x,10)
    assert torch.equal(c,d)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    z[1].fill_(1e8);changed,_=ddpm_sample(model,s,x,z=z)
    assert torch.equal(a,changed)
    one,_=ddim_sample(model,s,x,1)
    assert torch.equal(one,EpsilonParameterization(s).to_x0(model(x,torch.tensor(1000)),x,1000))


@pytest.mark.parametrize('K',[1,2,5,10,20,50,100,1000])
def test_tau_group5(K):
    tau=make_tau(1000,K)
    assert len(tau)==K and tau[-1]==1000 and tau[0]>=1
    assert all(a<b for a,b in zip(tau,tau[1:]))


def test_embedding_injection_group6():
    from src.models.diffusion.time_embedding import SinusoidalEmbedding
    e=SinusoidalEmbedding(128);t=torch.arange(1,1001)
    assert torch.equal(e(t),e(t)) and torch.unique(e(t),dim=0).shape[0]==1000
    model=EpsilonMLP({'width':16,'depth':6,'time_dim':128})
    out=model(torch.ones(8,2),torch.arange(1,9));out.sum().backward()
    assert len(model.time_projections)==6 and all(p.weight.grad is not None and p.weight.grad.abs().sum()>0 for p in model.time_projections)


def test_data_standardization_weights_and_quality():
    from src.diagnostics.sample_quality import frequencies,quality
    cfg=yaml.safe_load(Path('configs/lab7.yaml').read_text(encoding='utf-8'))
    roles,stats=roles_and_stats(cfg);z=standardize(roles['train'],stats)
    assert torch.allclose(z.mean(0),torch.zeros(2,dtype=torch.float64),atol=1e-12)
    assert torch.allclose(z.std(0,correction=0),torch.ones(2,dtype=torch.float64),atol=1e-12)
    assert torch.allclose(inverse_standardize(z,stats),roles['train'])
    assert not torch.equal(roles['train'][:5000],roles['test'])
    changed=copy.deepcopy(cfg);changed['data']['seeds']['test']=999
    assert roles_and_stats(changed)[1]==stats
    d=RareGMM(6,.1,.02);sample,labels=d.sample(100000,718,True)
    assert abs(float((labels==0).double().mean())-.02)<.002
    assert torch.equal(sample,d.sample(100000,718))
    assert d.mode_id(torch.tensor([[10.,10.]])).item()==6
    ref=frequencies(sample,d);q=quality(sample[:100],sample[:100],d,ref,1.)
    assert abs(q['MMD2'])<1e-12


def test_exact_diagnostic_self_error_zero():
    from src.diagnostics.timestep_error import timestep_errors
    s=schedule();e=ExactPredictor(RareGMM(6,.1,.02),{'mean':[0,0],'scale':[1,1]},s)
    g=torch.Generator().manual_seed(719);x=torch.randn(30,2,generator=g);n=torch.randn(30,2,generator=g)
    rows=timestep_errors(e,e,s,x,n,[1,50,1000])
    assert all(r['E_net']==0 for r in rows)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_runner_resume_and_unique_runs(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from src.engine.runner import ExperimentRunner
    cfg=yaml.safe_load(Path('configs/lab7.yaml').read_text(encoding='utf-8'))
    cfg['experiment']['purpose']='test';cfg['model']['width']=8;cfg['model']['time_dim']=8
    cfg['runtime']['device']=device;cfg['train']['max_steps']=6
    cfg['data']['test_train_size']=1024;cfg['logging']['run_root']=str(tmp_path/'runs')
    cfg['logging']['keep_epochs']=[]
    full=ExperimentRunner(copy.deepcopy(cfg)).run()
    part=ExperimentRunner(copy.deepcopy(cfg)).run(stop_after_epoch=1)
    resumed=ExperimentRunner(copy.deepcopy(cfg)).run(resume=part/'checkpoints/last.pt')
    assert len({full.name,part.name,resumed.name})==3
    a=torch.load(full/'checkpoints/last.pt',weights_only=False);b=torch.load(resumed/'checkpoints/last.pt',weights_only=False)
    def same(x,y):
        if isinstance(x,torch.Tensor):return torch.equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(i,j) for i,j in zip(x,y))
        return x==y
    for key in ['model','optimizer','scheduler','torch_rng','loader_rng','generative_state']:
        assert same(a[key],b[key]),key


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cpu_gpu_predictor_and_q_consistency():
    s=schedule();m=EpsilonMLP({'width':8,'depth':2,'time_dim':128}).double().eval()
    gpu=copy.deepcopy(m).cuda();x=torch.tensor([[.1,.2],[-.4,.6]],dtype=torch.float64);t=torch.tensor([1,1000])
    assert torch.allclose(m(x,t),gpu(x.cuda(),t.cuda()).cpu(),atol=1e-10,rtol=1e-8)
    a,_=q_sample(x,t,s,x);b,_=q_sample(x.cuda(),t.cuda(),s,x.cuda())
    assert torch.allclose(a,b.cpu(),atol=1e-12)
