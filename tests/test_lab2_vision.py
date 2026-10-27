import copy
import json
import numpy as np
import pytest
import torch
from src.config import load_config
from src.models.small_cnn import build_model
from src.models.vision.vit import patchify,unpatchify
from src.profiling.model_profiler import measure_macs,analytic_macs
from src.data.regimes import nested_subsets
from src.data.augmentations import augment_image,mix_batch
from src.diagnostics.corruptions import corrupt,LEVELS,CorruptedImages
from src.data.vision import ArrayImages
from src.engine.runner import ExperimentRunner
from src.analysis.learning_curves import paired_architecture,leader_reversal


@pytest.mark.parametrize('p',[2,4,8])
def test_L2_T01_patches_roundtrip(p):
    x=torch.randn(2,3,32,32);patches=patchify(x,p)
    assert patches.shape==(2,(32//p)**2,3*p*p)
    assert torch.equal(x,unpatchify(patches,p,32,32))


@pytest.mark.parametrize('name',['cifar_resnet18','cifar_vit'])
def test_L2_T02_T03_canonical_shapes_macs_immutable(name):
    torch.set_num_threads(2)
    cfg=load_config('configs/lab2.yaml')['model'];cfg['name']=name
    model=build_model(cfg).eval();before=copy.deepcopy(model.state_dict());x=torch.randn(2,3,32,32)
    with torch.no_grad():
        assert model(x).shape==(2,10)
        assert model.forward_features(x).shape==(2,512 if name=='cifar_resnet18' else 384)
        assert len(model.forward_features(x,return_intermediate=True))==(4 if name=='cifar_resnet18' else 6)
    measured=measure_macs(model,x)
    assert measured['MACs']==analytic_macs(cfg)['MACs']
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    if name=='cifar_vit':assert model.encoder.pos_embedding.shape==(1,65,384)


def test_L2_T03_single_convolution():
    model=torch.nn.Conv2d(3,7,3,padding=1)
    assert measure_macs(model,torch.randn(2,3,32,32))['MACs']==32*32*7*3*9


def test_L2_T04_nested_balanced():
    labels=np.repeat(np.arange(10),4300);fractions=[.05,.15,.30,.60,1.]
    subsets=nested_subsets(labels,list(range(43000)),fractions,987693601)
    again=nested_subsets(labels,list(range(43000)),fractions,987693601)
    assert subsets==again
    for r in fractions:
        assert len(subsets[r])==10*int(4300*r)
        assert np.bincount(labels[subsets[r]]).tolist()==[int(4300*r)]*10
    assert all(set(subsets[a])<set(subsets[b]) for a,b in zip(fractions[:-1],fractions[1:]))


def step_cfg(fixture_config,n=17):
    cfg=fixture_config
    cfg['experiment'].update(lab=2,purpose='test')
    cfg['data']['augmentation']='A1'
    cfg['train'].update(max_steps=5,epochs=100,validation_at_end_only=True)
    cfg['scheduler'].update(unit='optimizer_step',warmup_fraction=.05)
    cfg['logging'].update(save_every_steps=1000,console_every_steps=1000)
    return cfg


@pytest.mark.parametrize('n',[17,35,70])
def test_L2_T04_exact_updates_different_loader_lengths(fixture_config,monkeypatch,n):
    import src.engine.runner as runner
    original=runner.datasets
    def datasets(cfg):
        ds,info=original(cfg)
        ds['train']=torch.utils.data.TensorDataset(torch.randn(n,3,32,32),torch.arange(n)%10)
        return ds,info
    monkeypatch.setattr(runner,'datasets',datasets)
    path=ExperimentRunner(step_cfg(fixture_config)).run()
    summary=json.loads((path/'summary.json').read_text())
    assert summary['global_step']==5 and summary['status']=='COMPLETED'
    rows=[json.loads(l) for l in (path/'steps.jsonl').read_text().splitlines()]
    assert len(rows)==5 and rows[-1]['global_step']==5
    assert summary['validation_metric'] is not None


def test_L2_step_checkpoint_resume(fixture_config):
    cfg=step_cfg(fixture_config);cfg['train']['max_steps']=7
    gold=ExperimentRunner(cfg).run();part=ExperimentRunner(cfg).run(stop_after_epoch=1)
    resumed=ExperimentRunner(cfg).run(resume=part/'checkpoints/last.pt')
    ga=torch.load(gold/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    re=torch.load(resumed/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert ga['global_step']==re['global_step']==7
    assert all(torch.equal(v,re['model'][k]) for k,v in ga['model'].items())
    assert (gold/'steps.jsonl').read_text()==(resumed/'steps.jsonl').read_text()
    with pytest.raises(ValueError,match='partial'):
        ExperimentRunner(cfg).run(resume=gold/'checkpoints/last.pt')


def test_L2_T05_A0_deterministic_no_rng():
    x=torch.rand(3,32,32);rng=torch.get_rng_state().clone()
    assert torch.equal(augment_image(x,'A0',1),augment_image(x,'A0',2))
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('policy',['A1','randaugment+cutmix','trivialaugment+mixup'])
def test_augmentation_seed_and_label_mass(policy):
    x=torch.rand(3,32,32)
    assert torch.equal(augment_image(x,policy,14),augment_image(x,policy,14))
    if policy!='A1':
        batch=torch.stack([torch.zeros_like(x),torch.ones_like(x)])
        xx,yy,info=mix_batch(batch,torch.tensor([0,1]),policy)
        assert torch.allclose(yy.sum(1),torch.ones(2))
        assert torch.allclose(xx.mean((1,2,3)),yy[:,1],atol=1e-6)


@pytest.mark.parametrize('name',list(LEVELS))
def test_L2_T06_corruptions_identity_labels_seed(name):
    x=torch.rand(3,32,32)
    assert torch.equal(corrupt(x,name,0,123),x)
    for s in [1,2,3]:
        a=corrupt(x,name,s,123);b=corrupt(x,name,s,123)
        assert torch.equal(a,b) and a.shape==x.shape and a.min()>=0 and a.max()<=1
    base=ArrayImages(np.zeros((2,32,32,3),dtype=np.uint8),[2,7],[0,1],[0,0,0],[1,1,1])
    ds=CorruptedImages(base,name,1,123)
    assert ds[0][1]==2 and ds[1][1]==7


def test_corruption_content_semantics():
    x=torch.linspace(0,1,3072).reshape(3,32,32)
    contrast=corrupt(x,'contrast',1,4)
    assert torch.allclose(contrast.mean(),x.mean())
    assert torch.allclose(contrast-x.mean(),.6*(x-x.mean()),atol=1e-7)
    for s,d in enumerate([2,4,8],1):
        shifted=corrupt(torch.ones(3,32,32),'translation',s,123)
        assert shifted.count_nonzero()==3*32*(32-d)
    for s in [1,2,3]:
        assert torch.equal(corrupt(torch.ones(3,32,32)*.5,'blur',s,1),torch.ones(3,32,32)*.5) or torch.allclose(corrupt(torch.ones(3,32,32)*.5,'blur',s,1),torch.ones(3,32,32)*.5,atol=1e-6)
    shuffled=corrupt(x,'patch_shuffle',3,8)
    assert torch.equal(shuffled.flatten().sort().values,x.flatten().sort().values)


def test_L2_statistical_rules():
    assert paired_architecture([.8,.8,.8],[.7,.69,.71])['established']
    assert not paired_architecture([.8,.8,.8],[.7,.9,.71])['established']
    assert leader_reversal(.02,-.015) and not leader_reversal(.005,-.1)
