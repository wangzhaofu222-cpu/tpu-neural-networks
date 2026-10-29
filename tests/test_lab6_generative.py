import copy,inspect,math,json
from pathlib import Path
import numpy as np
import pytest
import torch,yaml
from torch import nn
from src.generative.distributions import distribution,make_roles
from src.generative.ebm import EnergyModel,energy_loss,tail_bound
from src.generative.samplers import langevin_sample,reflect,ReplayBuffer
from src.generative.flows import AffineCoupling,CoordinateSwap,NormalizingFlow
from src.diagnostics.invertibility import inverse_error,autograd_logdet
from src.diagnostics.mcmc import quadratic_chains,mode_switches
from src.diagnostics.density_metrics import mmd_squared,kernel_mean,sample_metrics,paired_statistics,select_flow_depth
from src.generative.density import density_grid,grid
from src.analysis.generative_comparison import match_parameters
from src.engine.runner import ExperimentRunner

@pytest.mark.parametrize('name',['two_blobs','eight_gaussians','four_corners','three_blobs'])
def test_distribution_roles_and_modes(name):
    d=distribution(name);assert torch.equal(d.mode_id(d.centers),torch.arange(len(d.centers)))
    assert d.mode_id(torch.tensor([[15.,15.]])).item()==-1
    a=make_roles(name);b=make_roles(name);assert set(a)=={'train','val','test'}
    assert [len(a[r]) for r in a]==[20000,5000,5000] and all(torch.equal(a[r],b[r]) for r in a)
    assert not torch.equal(a['val'],a['test'])
    assert d.mode_probabilities().sum()==pytest.approx(1.)

def test_true_density_midpoint_and_integral():
    d=distribution('two_blobs');assert float(d.log_prob(torch.zeros(1,2,dtype=torch.float64)))==pytest.approx(-21.65,abs=.01)
    assert float(d.log_prob(grid()).exp().sum()*.0625)==pytest.approx(1,abs=1e-5)

def test_mandatory1_energy_shape_and_external_bound():
    model=EnergyModel({'width':128});x=torch.randn(1000,2)*50
    e=model(x);assert e.shape==(1000,)
    q=x.square().sum(-1)/18;assert (e>=q-1-1e-5).all()
    # Avoid float32 cancellation when subtracting far-field quadratic values ~1e3.
    xd=x.double();md=model.double();qd=xd.square().sum(-1)/18
    assert torch.allclose(md(xd)-qd,md.residual(xd).squeeze(-1),atol=1e-5)
    assert tail_bound()<.01 and tail_bound()==pytest.approx(.0029,abs=.0001)

def test_mandatory2_noise_and_full_quadratic_variance():
    assert 'sigma' not in inspect.signature(langevin_sample).parameters
    r,traj,noise=quadratic_chains(.05)
    assert r['chains']==8 and r['draws_per_chain']==200000 and r['burn_in']==10000
    assert traj.shape==(8,210001) and noise.shape==(8,210000)
    assert r['theory_variance']==pytest.approx(1/(1-.05/2)) and r['relative_error']<=.05
    _,_,info=langevin_sample(lambda x:x.square().sum(-1)/2,torch.zeros(8,1),.05,1.,2,None,return_diagnostics=True)
    assert info['sigma_squared']==pytest.approx(2*.05)

def test_quadratic_finite_recurrence_matches_explicit_updates():
    r,traj,noise=quadratic_chains(.05,burn_in=3,draws=20)
    x=torch.zeros(8,dtype=torch.double)
    for k in range(23):
        x.requires_grad_(True);grad=torch.autograd.grad((x*x/2).sum(),x)[0]
        x=(x-.05*grad+r['sigma']*noise[:,k]).detach()
        assert torch.allclose(x,traj[:,k+1],atol=1e-14,rtol=1e-14)

def test_mandatory3_sampler_preserves_parameters_and_gradients():
    m=EnergyModel({'width':8});before={k:v.clone() for k,v in m.state_dict().items()}
    for p in m.parameters():p.grad=torch.ones_like(p)
    _,traj=langevin_sample(m,torch.randn(20,2),.02,1.,5,((-15,15),(-15,15)))
    assert traj.shape==(6,20,2) and all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in m.parameters())

def test_mandatory3_two_phase_gradient_and_wrong_detach_rejected():
    class LinearEnergy(nn.Module):
        def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor([.5,-.8],dtype=torch.double))
        def forward(self,x):return x@self.weight
    m=LinearEnergy();positive=torch.randn(40,2,dtype=torch.double);negative=torch.randn(40,2,dtype=torch.double,requires_grad=True)
    expected=positive.mean(0)-negative.detach().mean(0);loss=energy_loss(m,positive,negative);loss.backward()
    assert (m.weight.grad-expected).abs().max()<1e-5 and negative.grad is None
    wrong=torch.autograd.grad(m(positive).mean()-m(negative).mean().detach(),m.weight)[0]
    assert (wrong-expected).abs().max()>1e-5

def constant_layer(scale=-1.,shift=0.):
    m=AffineCoupling(8,4.).double()
    with torch.no_grad():
        m.raw_s[-1].weight.zero_();m.raw_s[-1].bias.fill_(4*math.atanh(scale/4))
        m.t[-1].weight.zero_();m.t[-1].bias.fill_(shift)
    return m

def test_mandatory4_random_roundtrip_and_signed_jacobian():
    torch.manual_seed(5);m=NormalizingFlow({'depth':4,'width':16,'cap':4.}).double()
    x=torch.randn(200,2,dtype=torch.double);x=x/x.norm(dim=-1,keepdim=True)*torch.rand(200,1,dtype=torch.double)*5
    assert inverse_error(m,x).median()<1e-5
    assert autograd_logdet(m,x[:20])['max_error']<1e-5
    layer=constant_layer();analytic=layer(x)[1];assert torch.allclose(analytic,-torch.ones_like(analytic),atol=1e-12)
    assert autograd_logdet(layer,x)['max_error']<1e-5
    random=AffineCoupling(16,4.).double();assert random.scale(x[:,:1]).abs().sum()>0
    assert autograd_logdet(random,x)['max_error']<1e-5

def test_mandatory5_identity_permutation_density_and_batch_independence():
    x=torch.randn(20,2,dtype=torch.double);layer=constant_layer(0.,0.)
    assert (layer(x)[0]-x).abs().max()<1e-6
    perm=CoordinateSwap();assert torch.equal(perm.inverse(perm(x)[0]),x) and torch.equal(perm(x)[1],torch.zeros(20,dtype=torch.double))
    m=NormalizingFlow({'depth':1,'width':8,'cap':4.}).double();m.layers[0]=constant_layer(-1.,0.)
    z,_=m(x);oracle=torch.distributions.Normal(0.,1.).log_prob(z).sum(-1)-1
    assert torch.allclose(m.log_prob(x),oracle,atol=1e-12)
    assert torch.allclose(m.log_prob(x[:1]),m.log_prob(x)[:1],atol=1e-12)

def test_mandatory6_grid_flag_and_bitwise_replay(tmp_path):
    m=EnergyModel({'width':8});r,data=density_grid(m,distribution('two_blobs'),'ebm')
    assert r['normalized_on_omega'] and r['grid_integral']==pytest.approx(1.,abs=1e-12)
    with pytest.raises(NotImplementedError):m.log_prob(torch.zeros(1,2))
    b=ReplayBuffer();b.sample(256);b.update(torch.randn(256,2));p=tmp_path/'replay.pt';torch.save(b.state_dict(),p)
    restored=ReplayBuffer(seed=8);restored.load_state_dict(torch.load(p,weights_only=False))
    assert torch.equal(b.points,restored.points) and torch.equal(b.sample(200),restored.sample(200))

def test_reflection_is_not_clipping_and_handles_multiple_crossings():
    x=torch.tensor([[-16.,16.],[47.,-47.]])
    assert torch.equal(reflect(x,((-15,15),(-15,15))),torch.tensor([[-14.,14.],[-13.,13.]]))
    assert not torch.equal(reflect(x,((-15,15),(-15,15))),x.clamp(-15,15))

def test_mmd_matches_direct_kernel_oracle_and_mode_outliers():
    x=torch.randn(15,2,dtype=torch.double);y=torch.randn(17,2,dtype=torch.double);h=.8
    oracle=lambda a,b:torch.exp(-((a[:,None,:]-b[None,:,:])**2).sum(-1)/(2*h*h)).mean()
    expected=oracle(x,x)+oracle(y,y)-2*oracle(x,y)
    assert mmd_squared(x,y,h)==pytest.approx(float(expected),abs=1e-12)
    assert mmd_squared(x,x,h)==pytest.approx(0.,abs=1e-12)
    d=distribution('two_blobs');ref=d.centers.repeat(20,1);out=torch.ones_like(ref)*15
    met=sample_metrics(out,ref,d,h);assert met['Cov']==0 and met['Dmode']==1 and met['mode_counts_including_outlier_first'][0]==len(out)

@pytest.mark.parametrize('depth',[2,4,8,16])
def test_capacity_matching(depth):
    r=match_parameters(depth,4.);assert r['relative_gap_over_smaller']<=.2 and r['flow_depth']==depth

def test_paired_statistics_noise_gate_and_val_depth_rule():
    r=paired_statistics([.5,.6,.7],[.1,.2,.3],.001);assert r['conclusion']=='различие установлено'
    assert paired_statistics([.1,.1,.1],[0.,0.,0.],.2)['conclusion']=='различие не установлено'
    rows=[{'depth':2},{'depth':4},{'depth':8},{'depth':16}]
    p={2:torch.ones(10)*3,4:torch.ones(10)*2,8:torch.ones(10)*2,16:torch.ones(10)}
    assert select_flow_depth(rows,p)['selected_depth']==4

def test_real_git_preregistration_gate():
    from src.utils.preregistration import verify_git_hypothesis
    assert verify_git_hypothesis('25b13ff0b8e3566aeb6ec0a718050462eae6abbe','experiments/lab5/hypothesis.md')
    with pytest.raises(ValueError):verify_git_hypothesis('', 'experiments/lab5/hypothesis.md')
    with pytest.raises(ValueError):verify_git_hypothesis('25b13ff','experiments/lab5/exploratory_hypotheses.md.not_present')

@pytest.mark.parametrize('kind',['energy_model','normalizing_flow'])
def test_shared_runner_resume_optimizer_rng_replay_and_unique_RUN(tmp_path,kind):
    c=yaml.safe_load(Path('configs/lab6.yaml').read_text(encoding='utf-8'))
    c['experiment']['purpose']='test';c['data']['test_train_size']=512;c['model'].update(name=kind,width=8,depth=2)
    c['train']['max_steps']=8;c['sampling']['steps']=3;c['logging'].update(run_root=str(tmp_path/'runs'),keep_epochs=[])
    full=ExperimentRunner(c).run();part=ExperimentRunner(c).run(stop_after_epoch=2);resumed=ExperimentRunner(c).run(resume=part/'checkpoints/last.pt')
    a=torch.load(full/'checkpoints/last.pt',weights_only=False);b=torch.load(resumed/'checkpoints/last.pt',weights_only=False)
    assert len({full.name,part.name,resumed.name})==3 and a['global_step']==b['global_step']==8
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert torch.equal(a['torch_rng'],b['torch_rng']) and torch.equal(a['loader_rng'],b['loader_rng'])
    for k,pa in a['optimizer']['state'].items():
        assert all(torch.equal(v,b['optimizer']['state'][k][name]) for name,v in pa.items())
    if kind=='energy_model':
        assert torch.equal(a['generative_state']['replay']['points'],b['generative_state']['replay']['points'])
        assert torch.equal(a['generative_state']['replay']['generator_state'],b['generative_state']['replay']['generator_state'])
