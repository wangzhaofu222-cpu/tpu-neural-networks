import copy
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from src.config import load_config,validate
from src.models.sequence.transformer import SequenceTransformer
from src.models.sequence.positional import Position,sinusoid,rotate
from src.data.synthetic_sequence import example,Sequences,KEY,MARKER,sequence_datasets
from src.diagnostics.attention_stats import attention_statistics
from src.diagnostics.extrapolation import paired,slopes
from src.metrics.objective import correct_count
from src.engine.runner import ExperimentRunner

def config(tmp_path):
    c=load_config('configs/lab3.yaml');c['experiment']['purpose']='test';c['runtime']['device']='cpu'
    c['data'].update(root=str(tmp_path/'data'),steps_per_epoch=2,evaluation_examples=4)
    c['train'].update(batch_size=2,max_steps=4);c['logging']['run_root']=str(tmp_path/'runs')
    c['model'].update(width=16,heads=2,ffn=32);c['logging']['save_every_steps']=1
    return c

def test_attention_rows_and_mask(tmp_path):
    c=config(tmp_path);m=SequenceTransformer(c['model']).eval();x=torch.randint(0,64,(2,16));mask=torch.ones(16,16,dtype=torch.bool).tril()
    _,a=m(x,mask=mask,return_attention=True)
    for v in a:
        assert (v.sum(-1)-1).abs().max()<=1e-5
        assert v.masked_select(~mask[None,None,:,:]).abs().max()==0

def test_no_position_permutation_equivariance(tmp_path):
    c=config(tmp_path);c['model']['position']='none';m=SequenceTransformer(c['model']).eval()
    x=torch.randint(0,64,(2,30));p=torch.randperm(30)
    assert (m(x[:,:,] )[:,:,p]-m(x[:,p])).abs().max()<1e-5

def test_sinusoid_extrapolation():
    p=torch.tensor([128,256,1024]);v=sinusoid(p,128)
    assert v.shape==(3,128)
    for row,m in enumerate(p.tolist()):
        for i in range(64):
            assert abs(v[row,2*i].item()-math.sin(m/10000**(2*i/128)))<1e-6
            assert abs(v[row,2*i+1].item()-math.cos(m/10000**(2*i/128)))<1e-6

def test_rope_shape_and_relative_identity():
    q=torch.randn(1,2,1,32,dtype=torch.float64);k=torch.randn_like(q)
    first=(rotate(q,torch.tensor([5]))*rotate(k,torch.tensor([9]))).sum(-1)
    second=(rotate(q,torch.tensor([105]))*rotate(k,torch.tensor([109]))).sum(-1)
    rhs=(q*rotate(k,torch.tensor([4]))).sum(-1)
    assert rotate(q,torch.tensor([5])).shape==q.shape
    assert (first-second).abs().max()<1e-5 and (first-rhs).abs().max()<1e-5

def test_absolute_rules():
    m=Position('absolute',16,128)
    assert torch.equal(m.absolute(128,'cpu'),m.table)
    p=m.absolute(256,'cpu');assert torch.equal(p[0],m.table[0]) and torch.equal(p[-1],m.table[-1])
    m.rule='P1';p=m.absolute(256,'cpu');assert torch.equal(p[:128],m.table)
    assert torch.equal(p,m.absolute(256,'cpu'))
    m.rule='P3'
    with pytest.raises(ValueError):m.absolute(129,'cpu')

@pytest.mark.parametrize('task',['copy','retrieval','needle'])
def test_generator_determinism_and_targets(task):
    a=example(task,127,np.random.default_rng(5));b=example(task,127,np.random.default_rng(5))
    assert all(torch.equal(x,y) for x,y in zip(a,b));x,y,t=a
    valid=y!=-100;assert torch.equal(y[valid],x[t[valid]])
    if task=='copy':assert len(x)==126 and (x[63:]==64).all()
    if task=='retrieval':
        keys=x[torch.where(x==KEY)[0]+1];assert len(keys)==len(set(keys.tolist()))
        assert (keys==x[-1]).sum()==1
    if task=='needle':assert (x==MARKER).sum()==1

def test_homogeneous_batches_and_test_independence(tmp_path):
    c=config(tmp_path);ds,info=sequence_datasets(c);before=[ds['train'][i] for i in range(4)]
    for l in [128,256,512,1024]:Sequences('copy',128,1,'test',200,length=l)[0]
    assert all(all(torch.equal(a,b) for a,b in zip(before[i],ds['train'][i])) for i in range(4))
    assert len(before[0][0])==len(before[1][0])
    assert all(64<=len(x[0])<=128 for x in before)
    assert Path(info['cal_path']).exists()
    with pytest.raises(PermissionError):Sequences('copy',128,1,'cal',2,length=128)[0]
    assert not torch.equal(Sequences('copy',128,1,'val',2,length=128)[0][0],Sequences('copy',128,1,'test',2,length=128)[0][0])

def test_target_only_loss_and_accuracy():
    from src.metrics.objective import objective
    logits=torch.randn(2,64,10,requires_grad=True);y=torch.full((2,10),-100);y[:,-1]=3
    objective(logits,y).backward();assert logits.grad[:,:,:-1].abs().max()==0
    assert correct_count(logits,y)[1]==2

def test_attention_uniform_numerical_stability():
    a=torch.ones(2,4,16,16)/16;t=torch.full((2,16),-1);t[:,-1]=3
    r=attention_statistics(a,t)
    assert (r['entropy']-math.log(16)).abs().max()<1e-5
    assert torch.isfinite(r['distance']).all();assert torch.nanmean(r['target_mass'])==1/16
    a.zero_();a[:,:,:,0]=1
    assert torch.isfinite(attention_statistics(a,t)['entropy']).all()

def test_paired_statistics_and_log_slopes():
    assert paired([.8,.81,.79],[.2,.21,.19])['established']
    assert not paired([.8,.1,.8],[.2,.8,.2])['established']
    assert not paired([.2]*3,[.2]*3)['established']
    s=slopes([64,128,256],[64**2,128**2,256**2]);assert abs(s['global']-2)<1e-10

def test_resume_exact_and_unique_run_ids(tmp_path):
    c=config(tmp_path);full=ExperimentRunner(c).run();partial=ExperimentRunner(c).run(stop_after_epoch=1)
    resumed=ExperimentRunner(c).run(resume=partial/'checkpoints/last.pt')
    assert len({full.name,partial.name,resumed.name})==3
    a=torch.load(full/'checkpoints/last.pt',weights_only=False);b=torch.load(resumed/'checkpoints/last.pt',weights_only=False)
    assert a['global_step']==b['global_step']==4
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    c['train']['max_steps']=8
    extended=ExperimentRunner(c).run(resume=full/'checkpoints/last.pt')
    assert json.loads((extended/'summary.json').read_text())['global_step']==8

def test_profile_schema_and_oom_recovery(tmp_path,monkeypatch):
    from src.profiling.sequence_profiler import measure,safe_measure
    c=config(tmp_path);m=SequenceTransformer(c['model']);r=measure(m,16,warmup=10,repeats=2)
    assert r['forward_seconds']>0 and r['training_step_seconds']>0 and r['peak_memory_bytes']>0
    assert r['tokens_per_second']==4*16/r['forward_seconds'] and r['backend']=='explicit'
    import src.profiling.sequence_profiler as module
    with monkeypatch.context() as patch:
        patch.setattr(module,'measure',lambda *a,**k:(_ for _ in ()).throw(torch.OutOfMemoryError('simulated OOM test only')))
        assert safe_measure(m,16)['status']=='OOM'
    assert measure(m,16,warmup=10,repeats=1)['status']=='COMPLETED_PROFILE'

def test_canonical_config_guard(tmp_path):
    c=load_config('configs/lab3.yaml');c['experiment']['purpose']='formal';validate(c)
    c['model']['depth']=4
    with pytest.raises(ValueError):validate(c)

def test_lr_selector_rejects_test_and_long_lengths():
    from src.diagnostics.extrapolation import select_learning_rate
    rows=[dict(role='val',length=128,position='sinusoidal',lr=lr,accuracy=.5) for lr in [.003,.001,.0003]]
    assert select_learning_rate(rows,128)['lr']==.0003
    for key,value in [('role','test'),('length',256)]:
        altered=copy.deepcopy(rows);altered[0][key]=value
        with pytest.raises(ValueError):select_learning_rate(altered,128)

def test_backbone_initialization_common_across_position_modes(tmp_path):
    c=config(tmp_path);states=[]
    for pe in ['none','absolute','sinusoidal','rope']:
        torch.manual_seed(1);c['model']['position']=pe
        states.append({k:v for k,v in SequenceTransformer(c['model']).state_dict().items() if not k.startswith('position.')})
    assert all(all(torch.equal(states[0][k],s[k]) for k in states[0]) for s in states)
