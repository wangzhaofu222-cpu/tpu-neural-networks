import numpy as np
import pytest
import torch
from torch import nn
from src.reliability.metrics import ece,classification_metrics
from src.reliability.calibration import TemperatureScaler
from src.reliability.uncertainty import uncertainty
from src.reliability.selective import risk_coverage,selective_risk,SelectivePredictor
from src.reliability.ood import detection_metrics
from src.corruptions.image import PARAMETERS,corrupt_record
from src.attacks import fgsm,pgd,NormalizedModel


@pytest.mark.parametrize('T',[.01,.5,1.,2.,100.])
def test_softmax_temperature(T):
    z=torch.randn(100,10,dtype=torch.float64);p=(z/T).softmax(1)
    assert torch.allclose(p.sum(1),torch.ones(100,dtype=torch.float64))
    assert torch.equal(p.argmax(1),z.argmax(1))
    if T==1:assert torch.equal(p,z.softmax(1))


@pytest.mark.parametrize('offset',[0.,.1])
def test_synthetic_ece(offset):
    g=np.random.default_rng(81);c=g.uniform(.2,1,100000);a=g.random(len(c))<c-offset
    assert abs(ece(c,a)['value']-offset)<=.01


@pytest.mark.parametrize('role,ids,ood',[('test',['test:1'],False),('val',['cal:1'],False),('val',['val:1'],True),('ood',['svhn:1'],True)])
def test_fit_rejects_roles(role,ids,ood):
    with pytest.raises(ValueError):TemperatureScaler(['val:1']).fit(torch.randn(1,3),[0],ids,role=role,ood=ood)


def test_temperature_fit():
    torch.manual_seed(4);z=torch.randn(100,3)*5;y=torch.arange(100)%3;ids=[f'val:{i}' for i in range(100)]
    c=TemperatureScaler(ids).fit(z,y,ids)
    assert c.fit_info['val_nll_after']<=c.fit_info['val_nll_before']
    assert torch.equal(c.transform(z).argmax(1),z.argmax(1))


def test_ood_rejects_metrics_labels():
    with pytest.raises(ValueError):classification_metrics(torch.ones(3,10),[0,1,2],ood=True)
    with pytest.raises(ValueError):detection_metrics([.1,.2,.3],[0,1,2])
    with pytest.raises(TypeError):detection_metrics([.1,.2],[False,True],labels=[0,1])


@pytest.mark.parametrize('kind',list(PARAMETERS))
@pytest.mark.parametrize('level',range(6))
def test_corruption_protocol(kind,level):
    x=torch.rand(3,32,32);a,y,i=corrupt_record(x,7,'test:2',kind,level,1);b,_,_=corrupt_record(x,7,'test:2',kind,level,1)
    assert torch.equal(a,b) and a.shape==x.shape and a.min()>=0 and a.max()<=1 and y==7 and i=='test:2'
    if level==0:assert torch.equal(a,x)


@pytest.mark.parametrize('name',['fgsm','pgd'])
@pytest.mark.parametrize('eps',[0.,2/255,8/255])
def test_attack_constraints_and_freeze(name,eps):
    torch.manual_seed(15);m=nn.Sequential(nn.Flatten(),nn.Linear(3*8*8,4)).eval();x=torch.rand(7,3,8,8);y=torch.arange(7)%4
    old={k:v.clone() for k,v in m.state_dict().items()}
    adv,info=fgsm(m,x,y,eps) if name=='fgsm' else pgd(m,x,y,eps,K=3,R=2)
    assert not m.training and not adv.requires_grad and all(p.grad is None for p in m.parameters())
    assert all(torch.equal(old[k],v) for k,v in m.state_dict().items())
    assert all(c['linf_max']<=eps+1e-6 and 0<=c['pixel_min']<=c['pixel_max']<=1 for c in info['constraints'])
    if eps==0:assert torch.equal(adv,x)
    if name=='pgd':assert torch.equal(info['selected_losses'],info['restart_losses'].max(0).values)


def test_normalization_wrapper():
    m=nn.Sequential(nn.Flatten(),nn.Linear(12,3)).eval();x=torch.rand(4,3,2,2);mean=[.4,.5,.6];std=[.2,.3,.4];w=NormalizedModel(m,mean,std)
    assert torch.equal(w(x),m((x-w.mean)/w.std))


def test_ece_boundaries_and_brier():
    r=ece([0.,.5,1.],[0,1,1],M=2);assert [b['count'] for b in r['bins']]==[1,2]
    z=torch.zeros(3,3);m=classification_metrics(z,[0,1,2]);assert abs(m['brier']-2/3)<1e-12
    assert ece([.9],[1],M=15)['empty_bins']==14


@pytest.mark.parametrize('binning',['equal_width','equal_count'])
def test_ece_schema(binning):
    r=ece(np.linspace(0,1,43),np.ones(43),M=15,binning=binning);assert sum(x['count'] for x in r['bins'])==43


def test_selective_oracle_and_empty():
    g=np.random.default_rng(51);u=g.random(50);c=g.random(50)>.3;r=risk_coverage(u,c)
    assert r['oracle_AURC']<=r['AURC'] and r['excess_AURC']>=0
    assert selective_risk(u,c,float('inf'))['risk']==float((~c).mean())
    assert selective_risk(u,c,-1)['risk'] is None
    assert r['risk'][-1]==float((~c).mean())


def test_ood_monotone_and_ties():
    u=np.array([0.,1.,1.,2.]);y=np.array([False,False,True,True]);a=detection_metrics(u,y);b=detection_metrics(np.exp(u),y)
    assert a['AUROC']==b['AUROC']==.875


def test_uncertainty_extremes():
    u=uncertainty(torch.tensor([[1e4,-1e4,0.],[0.,0.,0.]]))
    assert all(torch.isfinite(v).all() for v in u.values())
    assert all(v[0]<v[1] for v in u.values())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cpu_gpu_metrics():
    z=torch.randn(64,10);y=torch.arange(64)%10
    # Metrics deliberately operate in CPU float64 for one reproducible definition.
    assert classification_metrics(z,y)==classification_metrics(z.cuda().cpu(),y)


def test_paired_teacher_statistics():
    from reports.tools.lab8_common import pair_statistics
    assert pair_statistics([.1,.11,.09])['expected_positive_supported']
    assert pair_statistics([-.1,-.11,-.09])['conclusion']=='обратный вывод установлен'
    assert pair_statistics([.1,-.1,.02])['conclusion']=='различие переноса не установлено'


def test_cal_role_rejected():
    from reports.tools.lab8_common import role_data
    with pytest.raises(ValueError):role_data('cal')
