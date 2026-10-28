import copy
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from src.config import load_config,validate
from src.ssl.augmentations import build_augmentation,augmentation_hash
from src.ssl.views import Views
from src.ssl.losses import nt_xent
from src.ssl.simclr import SimCLR
from src.diagnostics.representation import representation_statistics
from src.evaluation.features import extract_arrays,state_digest
from src.evaluation.knn import weighted_knn
from src.analysis.label_efficiency import label_subsets,paired_gain,classify_ablation
from src.engine.runner import ExperimentRunner

def config(tmp_path):
    c=load_config('configs/lab4.yaml');c['experiment']['purpose']='test';c['runtime']['device']='cpu'
    c['data'].update(name='ssl_test_fixture',num_workers=0);c['train'].update(epochs=2,batch_size=2,amp=False)
    c['representation']['enabled']=False;c['logging'].update(run_root=str(tmp_path/'runs'),keep_epochs=[1,2])
    c['scheduler']['warmup_epochs']=0
    return c

@pytest.mark.parametrize('workers',[0,2])
def test_independent_views_and_workers(workers):
    images=np.random.default_rng(2).integers(0,256,(8,32,32,3),dtype=np.uint8)
    c=load_config('configs/lab4.yaml');ds=Views(images,range(8),[.5]*3,[.2]*3,c['augmentations'],4)
    x,y=next(iter(torch.utils.data.DataLoader(ds,batch_size=4,num_workers=workers)))
    assert x.shape==(4,2,3,32,32) and not torch.equal(x[:,0],x[:,1])
    assert not hasattr(ds,'labels')
    ds=Views(images,range(8),[.5]*3,[.2]*3,[],4);a,b,_=ds.two_views(0)
    assert torch.equal(a,b)

def test_augmentation_hash_and_all_five_operations():
    from PIL import Image
    image=Image.fromarray(np.random.default_rng(3).integers(0,256,(32,32,3),dtype=np.uint8))
    ops=load_config('configs/lab4.yaml')['augmentations']+[{'name':'grayscale','p':.2},{'name':'blur','p':.5}]
    assert augmentation_hash(ops)==augmentation_hash(copy.deepcopy(ops))
    assert augmentation_hash(ops)!=augmentation_hash(ops[:-1])
    assert build_augmentation(ops,[.5]*3,[.2]*3)(image).shape==(3,32,32)

@pytest.mark.parametrize('tau',[.1,.5,.01])
def test_nt_xent_known_values_and_stability(tau):
    n=8;a=torch.ones(n,16,dtype=torch.float64)
    loss,stats=nt_xent(a,a,tau);assert abs(loss.item()-math.log(2*n-1))<1e-5
    a=torch.eye(n,dtype=torch.float64);loss,_=nt_xent(a,a,tau)
    expected=math.log1p((2*n-2)*math.exp(-1/tau));assert abs(loss.item()-expected)<1e-5
    assert stats['negative_count']==2*n-2 and torch.isfinite(loss)

def test_equation_4_6_anchor_gradient():
    torch.manual_seed(5);anchor=torch.nn.functional.normalize(torch.randn(16,dtype=torch.float64),dim=0).requires_grad_()
    keys=torch.nn.functional.normalize(torch.randn(7,16,dtype=torch.float64),dim=1);tau=.1;positive=3
    scores=keys@anchor/tau;loss=torch.logsumexp(scores,0)-scores[positive]
    gradient=torch.autograd.grad(loss,anchor)[0];expected=(scores.softmax(0)@keys-keys[positive])/tau
    assert (gradient-expected).abs().max()<1e-5

def test_nt_xent_positives_permutation_and_gradients():
    torch.manual_seed(8);a=torch.randn(8,16,dtype=torch.float64,requires_grad=True);b=a.detach().clone().requires_grad_()
    loss,stats=nt_xent(a,b,.1);assert stats['positive_similarity']> .99999
    loss.backward();assert torch.isfinite(a.grad).all()
    p=torch.randperm(8);other,_=nt_xent(a.detach(),b.detach()[p],.1);assert other>loss

def test_effective_rank_known_cases_and_scale_convention():
    x=torch.arange(100).double()[:,None]*torch.ones(1,8,dtype=torch.float64)
    assert abs(representation_statistics(x)['effective_rank']-1)<1e-5
    torch.manual_seed(2);s=representation_statistics(torch.randn(20*16,16));assert s['effective_rank']>=.9*16
    s=representation_statistics(torch.ones(100,16));assert s['effective_rank'] is None and s['collapse_scale']
    x=torch.nn.functional.normalize(torch.randn(200,16,dtype=torch.float64),dim=1)
    s=representation_statistics(x,True);assert s['identity_4_9_error']<1e-5

def test_knn_reference_roles_and_equation():
    train=torch.eye(10);labels=torch.arange(10)
    assert torch.equal(weighted_knn(train,labels,train,k=3),labels)
    with pytest.raises(PermissionError):weighted_knn(train,labels,train,k=3,reference_role='test')

def test_label_stratification_nested_pairing_and_cal_guard():
    from src.data.ssl_data import role_indices
    labels=np.tile(np.arange(10),100);fractions=[.02,.05,.25,1.]
    for seed in [1,2,3]:
        subsets=label_subsets(labels,range(1000),fractions,seed)
        for r in fractions:
            assert np.bincount(labels[subsets[str(r)]],minlength=10).tolist()==[int(100*r)]*10
    with pytest.raises(PermissionError):role_indices({'roles':{'cal':[1,2]}},'cal')
    assert paired_gain([.8,.7,.9],[.2,.2,.2])['established']
    assert not paired_gain([.2,.3,.1],[.2,.2,.2])['established']
    assert classify_ablation(-1,2,False)=='indistinguishable_from_noise_single_seed'

def test_feature_repeatability_and_batchnorm_unchanged(tmp_path):
    m=SimCLR(config(tmp_path)['model']);m.train();before=state_digest(m)
    images=np.random.default_rng(3).integers(0,256,(4,32,32,3),dtype=np.uint8)
    a,b=extract_arrays(m,images,list(range(4)),[.5]*3,[.2]*3,'cpu',2)
    c,d=extract_arrays(m,images,list(range(4)),[.5]*3,[.2]*3,'cpu',2)
    assert (a-c).abs().max()<1e-6 and (b-d).abs().max()<1e-6 and state_digest(m)==before and m.training

def test_shared_runner_ssl_resume_and_rng_exact(tmp_path):
    c=config(tmp_path);full=ExperimentRunner(c).run();partial=ExperimentRunner(c).run(stop_after_epoch=1)
    resumed=ExperimentRunner(c).run(resume=partial/'checkpoints/last.pt')
    a=torch.load(full/'checkpoints/last.pt',weights_only=False);b=torch.load(resumed/'checkpoints/last.pt',weights_only=False)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert torch.equal(a['loader_rng'],b['loader_rng']) and torch.equal(a['torch_rng'],b['torch_rng'])
    assert len({full.name,partial.name,resumed.name})==3

def test_frozen_encoder_probe_full_state_unchanged(tmp_path):
    m=SimCLR(config(tmp_path)['model']);before={k:v.clone() for k,v in m.state_dict().items()}
    cache={'fingerprint':'fixture','split_fingerprint':'fixture','encoder_RUN_ID':'fixture',
        'train':{'role':'train','h':torch.randn(20,512),'labels':torch.arange(20)%10,'indices':list(range(20)),'encoder_state_sha256':state_digest(m)},
        'val':{'role':'val','h':torch.randn(10,512),'labels':torch.arange(10),'indices':list(range(10))}}
    f=tmp_path/'features.pt';torch.save(cache,f)
    c=config(tmp_path);c['data'].update(name='ssl_features',feature_path=str(f),feature_kind='h',label_indices=list(range(20)))
    c['model']={'name':'linear_probe','input_dim':512,'classes':10};ExperimentRunner(c).run()
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())

def test_unlabelled_path_never_accesses_labels(tmp_path,monkeypatch):
    import src.data.ssl_data as module
    from src.data.splits import stratified_split
    from src.utils.io import write_json
    images=np.random.default_rng(6).integers(0,256,(40,32,32,3),dtype=np.uint8)
    class UnlabelledOnly:
        data=images
        @property
        def targets(self):raise AssertionError('SSL accessed labels')
    monkeypatch.setattr(module,'CIFAR10',lambda *a,**k:UnlabelledOnly())
    split=stratified_split(np.arange(40)%10,1,1,1,10);sp=tmp_path/'split.json';write_json(sp,split)
    c=config(tmp_path);c['data'].update(name='ssl_cifar10',root=str(tmp_path),split_path=str(sp))
    ds,info=module.datasets(c);assert not info['pretraining_label_access'] and len(ds['train'])==20

def test_formal_budget_guards(tmp_path):
    c=load_config('configs/lab4.yaml');c['experiment']['purpose']='formal';validate(c)
    c['train']['batch_size']=32
    with pytest.raises(ValueError):validate(c)
