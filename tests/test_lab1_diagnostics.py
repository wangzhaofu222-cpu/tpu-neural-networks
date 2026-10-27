import copy
import math
import pytest
import torch
from src.models.small_cnn import build_model
from src.models.residual_mlp import ResidualBlock
from src.diagnostics.activations import ActivationMonitor, statistics
from src.diagnostics.gradients import gradient_statistics
from src.diagnostics.trainability import summarize, probe, chi_estimates


def config(**kw):
    return dict(name='deep_mlp', width=128, depth=4, classes=10, input_features=128,
                activation='relu', normalization='none', initialization='he', sigma=.02, alpha=1, **kw)


def test_monitor_preserves_output():
    model = build_model(config())
    x = torch.randn(64,128)
    expected = model(x)
    monitor = ActivationMonitor(model.diagnostic_layers())
    assert torch.equal(expected, model(x))
    monitor.close()


def test_monitor_preserves_gradients():
    a = build_model(config())
    b = copy.deepcopy(a)
    x = torch.randn(64,128)
    a(x).square().mean().backward()
    monitor = ActivationMonitor(b.diagnostic_layers())
    b(x).square().mean().backward()
    assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
    assert 'delta' in monitor.rows['h1']
    monitor.close()


@pytest.mark.parametrize('scheme', ['he','xavier','gaussian'])
@pytest.mark.parametrize('name', ['deep_mlp','residual_mlp'])
def test_initialization_variance(scheme,name):
    cfg = config()
    cfg.update(initialization=scheme,name=name)
    model = build_model(cfg)
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            assert torch.count_nonzero(layer.bias) == 0
            outn, inn = layer.weight.shape
            target = {'he':2/inn, 'xavier':2/(inn+outn), 'gaussian':.02**2}[scheme]
            if inn*outn >= 10000:
                assert abs(layer.weight.var().item()/target-1) <= .1


def test_zero_branch_identity():
    block = ResidualBlock(128)
    with torch.no_grad():
        for p in block.branch.parameters():
            p.zero_()
    x = torch.randn(64,128)
    assert (block(x)-x).abs().max() <= 1e-6


@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_numerical_failure_flag(bad):
    assert summarize({'h1':statistics(torch.tensor([bad]))}, [])['numerical_failure']
    assert summarize({}, [{'l':{'g':None,'numerical_failure':True}}])['numerical_failure']
    assert summarize({}, [], bad)['numerical_failure']


def test_float64_before_square():
    row = statistics(torch.tensor([1e25,-1e25]))
    assert math.isfinite(row['q']) and row['q'] > 1e49


def test_tanh_saturation():
    assert statistics(torch.tensor([0., .99]), 'tanh')['saturated_fraction'] == .5


def test_gi_median_per_layer():
    samples = [{'a':{'g':a},'b':{'g':b}} for a,b in [(1,100),(100,1),(2,2)]]
    assert summarize({},samples)['GI'] == 0


@pytest.mark.parametrize('norm',['batchnorm','layernorm'])
def test_probe_preserves_state_rng_and_gradients(norm):
    cfg = config()
    cfg['normalization'] = norm
    model = build_model(cfg).train()
    x = torch.randn(64,128)
    model(x).sum().backward()
    state = copy.deepcopy(model.state_dict())
    grads = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    result = probe(model,x,123,'relu')
    assert model.training
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),grads))
    assert result['gradients']['head']['g'] is None


def test_chi_effective_depth():
    rows = {'h1':{'q':1.,'delta':{'q':.25}},'h2':{'q':.5,'delta':{'q':.5}},
            'h3':{'q':1e-40,'delta':{'q':1.}}}
    result = chi_estimates(rows)
    assert result['fwd'] == .5 and result['bwd'] == .5
    assert result['last_layer'] == 'h2'


def test_shared_runner_lab1_summary(fixture_config, tmp_path):
    cfg = fixture_config
    from src.engine.runner import ExperimentRunner
    cfg['experiment'].update(lab=1,purpose='test')
    cfg['model'] = config()
    cfg['model']['input_features'] = 3072
    cfg['train']['epochs'] = 1
    cfg['diagnostics'] = {'enabled':True,'probe_batch_size':64,'loss_subset_size':1024}
    path = ExperimentRunner(cfg).run()
    import json
    summary = json.loads((path/'summary.json').read_text())
    assert summary['status'] == 'COMPLETED'
    assert summary['trainability']['rho_T'] > 0
    assert summary['trainability']['GI'] >= 0
    assert (path/'diagnostics.jsonl').exists()
    assert (path/'probe.final.json').exists()
