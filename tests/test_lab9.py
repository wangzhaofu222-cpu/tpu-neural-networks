import copy
import numpy as np
import pytest
import torch
from src.multimodal.alignment import alignment_windows,masked_mean
from src.multimodal.modality_mask import apply_mask,sample_modality_mask
from src.multimodal.fusion import FusionClassifier,late_probabilities,matching_width,parameter_count
from src.multimodal.pairs import derangement,make_pairs,require_defined_metric
from src.multimodal.corruptions import add_noise,time_mask,frame_dropout
from src.multimodal.audio import log_mel
from src.multimodal.identity import validate_clip_identity
from src.multimodal.cache import load_feature_cache
from src.models.audio_cnn import AudioCNN
from src.diagnostics.contribution import contribution,classification,majority_baseline


def cfg(mode='gated',gate='scalar'):
    return {'strategy':mode,'gate':gate,'classes':8,'visual_dim':7,'audio_dim':5,'hidden_width':9,'fusion_dim':6,'p_drop':0.}


def test_identity_split_frames():
    c={'clip_id':'a','actor_id':1,'emotion':0,'role':'train','audio_clip_id':'a','visual_clip_id':'a','frames':[{'clip_id':'a','actor_id':1,'role':'train'}]}
    assert validate_clip_identity([c],{'train':[1],'val':[2],'cal':[3],'test':[4]})
    for bad in [{**c,'audio_clip_id':'b'},{**c,'frames':[{'clip_id':'a','actor_id':1,'role':'test'}]}]:
        with pytest.raises(ValueError):validate_clip_identity([bad],{'train':[1],'val':[2]})
    with pytest.raises(ValueError):validate_clip_identity([c],{'train':[1],'val':[1]})


@pytest.mark.parametrize('r',[1,2,4])
@pytest.mark.parametrize('offset',[-250,0,62.5,250])
def test_alignment_no_gaps_overlap(r,offset):
    result=alignment_windows(range(18),351,hop=160,r=r,av_offset_ms=offset)
    assigned=result['assigned_indices'];times=result['audio_sample_times'];inside=(2*times>=-3200)&(2*(times-17*3200)<3200)
    assert len(assigned)==len(torch.unique(assigned)) and torch.equal(assigned,torch.where(inside)[0])


def test_alignment_half_open_and_padding():
    a=alignment_windows([0,1,2],60);assert 10 not in a['windows'][0] and 10 in a['windows'][1]
    valid=torch.arange(60)<31;b=alignment_windows([0,1,2],60,valid_audio=valid);assert (b['assigned_indices']<31).all()
    x=torch.tensor([[[2.],[4.],[999.]]]);assert masked_mean(x,[[1,1,0]]).item()==3.
    with pytest.raises(ValueError):masked_mean(x,[[0,0,0]])


@pytest.mark.parametrize('mode',['video','audio','early','gated'])
@pytest.mark.parametrize('gate',['scalar','vector'])
def test_mask_present_modality_invariance(mode,gate):
    torch.manual_seed(8);m=FusionClassifier(cfg(mode,gate)).eval();v=torch.randn(5,7);a=torch.randn(5,5)
    p=m(visual=v,audio=a,mask=(1,0));q=m(visual=v,audio=torch.randn_like(a)*100,mask=(1,0));assert torch.equal(p,q)
    vv,aa=apply_mask(v,a,(1,0));assert torch.equal(vv,v) and torch.equal(aa,torch.zeros_like(a))
    with pytest.raises(ValueError):m(visual=v,audio=a,mask=(0,0))
    assert torch.allclose(p.sum(1),torch.ones(5))


@pytest.mark.parametrize('mask',[(2,0),(.5,1),(-1,1),(0,0)])
def test_illegal_masks(mask):
    with pytest.raises(ValueError):apply_mask(torch.ones(2,7),torch.ones(2,5),mask)


@pytest.mark.parametrize('g',[0.,1.])
def test_gate_limits(g):
    m=FusionClassifier(cfg()).eval();v=torch.randn(5,7);a=torch.randn(5,5)
    p=m(visual=v,audio=a,forced_gate=g)
    z=torch.tanh(m.video_projection(v)) if g else torch.tanh(m.audio_projection(a))
    assert torch.allclose(p,m.head(z).softmax(1))


@pytest.mark.parametrize('weight',[0.,.5,1.])
def test_late_limits(weight):
    v=torch.randn(4,8).softmax(1);a=torch.randn(4,8).softmax(1);p=late_probabilities(v,a,weight)
    assert torch.allclose(p.sum(1),torch.ones(4))
    if weight==0:assert torch.equal(p,a)
    if weight==1:assert torch.equal(p,v)
    assert torch.equal(late_probabilities(v,a,weight,mask=(1,0)),v)
    assert torch.equal(late_probabilities(v,a,weight,mask=(0,1)),a)


def test_no_audio_contribution_and_derangement():
    m=FusionClassifier(cfg('video')).eval();v=torch.randn(16,7);a=torch.randn(16,5);y=np.arange(16)%8
    full=m(visual=v,audio=a).detach().numpy();without=m(visual=v,audio=a,mask=(1,0)).detach().numpy();nov=m(visual=v,audio=a,mask=(0,1)).detach().numpy()
    out=contribution(full,without,nov,y,.03);assert out['Ra']==0 and np.all(np.array(out['delta_a'])==0)
    p=derangement(16,81);assert set(p)==set(range(16)) and np.all(p!=np.arange(16));assert np.array_equal(np.sort(y[p]),np.sort(y))
    assert np.array_equal(p,derangement(16,81))


@pytest.mark.parametrize('snr',[30,20,10,0,-5])
def test_snr_measured(snr):
    x=torch.linspace(-.9,.8,56000);y,info=add_noise(x,snr,1)
    assert abs(info['measured_snr_db']-snr)<=.1 and torch.equal(y,add_noise(x,snr,1)[0])


def test_silence_and_level0():
    with pytest.raises(ValueError):add_noise(torch.zeros(1000),20)
    x=torch.randn(64,351);assert torch.equal(time_mask(x,0),x)
    v=torch.tensor([True,True,False]);assert torch.equal(frame_dropout(v,0),v)


@pytest.mark.parametrize('metric',['accuracy','macro_f1','ece','risk','selective_risk','Macro-F1','selective-risk','F1'])
def test_protocol_b_undefined(metric):
    with pytest.raises(ValueError,match='no unique true emotion'):require_defined_metric('B',metric)


@pytest.mark.parametrize('protocol',['A','B'])
def test_different_actor_pairs(protocol):
    clips=[{'clip_id':f'{i}_{j}_{rep}','actor_id':i,'emotion':j} for i in range(5) for j in range(8) for rep in range(2)]
    pairs=make_pairs(clips,protocol,count=200,seed=1);assert len(pairs)==200
    assert all(p['video_actor_id']!=p['audio_actor_id'] and ((p['video_emotion']==p['audio_emotion'])==(protocol=='A')) for p in pairs)
    assert pairs==make_pairs(clips,protocol,count=200,seed=1)


def test_log_mel_351_and_audio_padding():
    m=log_mel(torch.zeros(56000));assert m.shape==(64,351) and torch.isfinite(m).all()
    torch.manual_seed(9);model=AudioCNN({'classes':8}).eval();x=torch.randn(2,2,64,51);x[:,1]=1
    padded=torch.nn.functional.pad(x,(0,20));assert torch.allclose(model(x),model(padded),atol=1e-6)


def test_missing_empty_video():
    m=FusionClassifier(cfg()).eval();v=torch.empty(2,0,7);a=torch.randn(2,3,5);meta={'visual_valid':torch.empty(2,0,dtype=torch.bool),'audio_valid':torch.ones(2,3,dtype=torch.bool)}
    p=m(visual=v,audio=a,mask=(0,1),align=meta);assert torch.isfinite(p).all()
    with pytest.raises(ValueError):m(visual=v,audio=a,mask=(1,1),align=meta)


def test_dropout_probabilities_and_parameter_matching():
    g=torch.Generator().manual_seed(91);m=sample_modality_mask(100000,.5,generator=g);assert (m.sum(1)>0).all()
    for mask,p in [([1,1],.5),([1,0],.25),([0,1],.25)]:assert abs(float((m==torch.tensor(mask)).all(1).float().mean())-p)<.01
    target=parameter_count(FusionClassifier(cfg()));match=matching_width(7,8,target)
    z=cfg('video');z['hidden_width']=match['width'];assert parameter_count(FusionClassifier(z))==match['actual']


def test_cache_checkpoint_identity(tmp_path):
    p=tmp_path/'cache.pt';torch.save({'clip_id':'a','encoder_sha256':'correct','dataset_hash':'d','features':torch.ones(2,4)},p)
    assert load_feature_cache(p,encoder_sha256='correct',dataset_hash='d',clip_id='a')['features'].shape==(2,4)
    with pytest.raises(ValueError):load_feature_cache(p,encoder_sha256='wrong',dataset_hash='d',clip_id='a')


def test_majority_real_baseline_and_undefined():
    y=np.repeat(np.arange(8),[20]+[40]*7);b=majority_baseline(y,y);assert abs(b['macro_f1']-1/34)<1e-12
    p=np.eye(8)[y];out=contribution(p,p,p,y,1.);assert out['Rtilde_a'] is None and out['undefined']
