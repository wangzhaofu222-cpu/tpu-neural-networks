"""Shared-harness integration: resume, macro-F1 selection and role isolation."""
import copy,json
from pathlib import Path
import torch,yaml
import pytest
from src.engine.runner import ExperimentRunner
from src.utils.io import write_json

def test_shared_harness_resume_and_frozen_role_isolation(tmp_path):
    cfg=yaml.safe_load(Path('configs/lab9.yaml').read_text());cfg['experiment'].update(purpose='test',name='lab9_fixture')
    cfg['logging'].update(run_root=str(tmp_path/'runs'));cfg['train'].update(epochs=3,batch_size=8)
    cfg['model'].update(hidden_width=7,fusion_dim=5)
    manifest=tmp_path/'manifest.json';write_json(manifest,{'status':'PASS','synthetic_fixture':True,'clips':[{'role':'train'}]*24+[{'role':'val'}]*8,'raw_identity_hash':'unit_fixture','split_fingerprint':'unit_fixture'})
    cfg['data']['manifest_path']=str(manifest);cfg['data']['feature_path']=str(tmp_path/'features.pt')
    g=torch.Generator().manual_seed(92);data={r:{'x':torch.randn(n,642,generator=g),'y':torch.arange(n)%8} for r,n in [('train',24),('val',8)]}
    for r in data:data[r]['x'][:,-2:]=1
    # Deliberately omit test/cal: shared training must not access either.
    torch.save(data,cfg['data']['feature_path'])
    full=ExperimentRunner(copy.deepcopy(cfg)).run();part=ExperimentRunner(copy.deepcopy(cfg)).run(stop_after_epoch=1)
    resumed=ExperimentRunner(copy.deepcopy(cfg)).run(resume=part/'checkpoints/last.pt')
    a=torch.load(full/'checkpoints/last.pt',weights_only=False);b=torch.load(resumed/'checkpoints/last.pt',weights_only=False)
    assert a['global_step']==b['global_step']==9
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert a['best_metric_name']=='macro_f1'
    assert json.loads((full/'manifest.json').read_text())['cal_read'] is False
    assert (full/'checkpoints/best.pt').exists()

def test_frozen_encoder_no_gradients():
    from src.models.audio_cnn import AudioCNN
    from src.multimodal.fusion import FusionClassifier
    cfg=yaml.safe_load(Path('configs/lab9.yaml').read_text())['model'];audio=AudioCNN({'classes':8}).requires_grad_(False).eval()
    x=torch.randn(2,2,64,31);x[:,1]=1
    with torch.no_grad():z,_=audio.forward_features(x)
    model=FusionClassifier(cfg);loss=model(torch.cat([torch.randn(2,512),z.mean(1)],1)).square().mean();loss.backward()
    assert all(p.grad is None for p in audio.parameters())
    assert all(p.grad is not None for p in model.parameters())

def test_audio_shared_harness_variable_length_padding(tmp_path):
    cfg=yaml.safe_load(Path('configs/lab9.yaml').read_text());cfg['experiment'].update(purpose='test',name='lab9_audio_fixture');cfg['logging']['run_root']=str(tmp_path/'runs');cfg['train'].update(epochs=2,batch_size=2,early_stopping_patience=0);cfg['model']={'name':'audio_cnn','classes':8};cfg['data']['name']='ravdess_audio'
    clips=[]
    for role in ['train','val']:
        for i,n in enumerate([31,45]):
            p=tmp_path/f'{role}_{i}.pt';torch.save(torch.randn(64,n),p);clips.append({'role':role,'log_mel_path':str(p),'emotion':i})
    manifest=tmp_path/'manifest.json';write_json(manifest,{'status':'PASS','synthetic_fixture':True,'clips':clips,'raw_identity_hash':'fixture','split_fingerprint':'fixture'});stats=tmp_path/'stats.json';write_json(stats,{'mean':[0.]*64,'std':[1.]*64});cfg['data'].update(manifest_path=str(manifest),normalization_path=str(stats))
    path=ExperimentRunner(cfg).run();s=torch.load(path/'checkpoints/best.pt',weights_only=False)
    assert s['best_metric_name']=='macro_f1' and s['global_step']>=1
    assert json.loads((path/'manifest.json').read_text())['synthetic_fixture'] is True

def test_conflict_auroc_orientation_and_ties():
    from reports.tools.lab9_e3 import auroc
    assert auroc([0,0,1,1],[.1,.2,.8,.9])==1
    assert auroc([0,0,1,1],[.9,.8,.2,.1])==0
    assert auroc([0,0,1,1],[1,1,1,1])==.5

def test_capacity_survives_inference_freeze():
    from reports.tools.lab9_common import training_capacity,matched_control
    from src.multimodal.fusion import FusionClassifier,parameter_count
    cfg=yaml.safe_load(Path('configs/lab9.yaml').read_text())['model'];m=FusionClassifier(cfg);expected=parameter_count(m);m.requires_grad_(False)
    counts=training_capacity(m);assert counts['total_trainable']==expected>0
    assert counts['fusion_trainable']+counts['head_trainable']==expected
    assert all(not p.requires_grad for p in m.parameters())
    assert matched_control(expected,'video')['relative_difference']<=.01

@pytest.mark.parametrize('indices',[[0,2],[0,1]])
def test_canonical_windows_preserve_missing_visual_slots(tmp_path,indices):
    from reports.tools.lab9_alignment_metadata import complete_alignment
    p=tmp_path/'video.pt';torch.save({'canonical_video_count':3},p);c={'clip_id':'fixture','video_frames_path':str(p),'av_offset_ms':0,'frames':[{'raw10fps_index':2,'face_found':False}]}
    result=complete_alignment(c,{'indices':indices},{'z':torch.zeros(60,128)})
    assert len(result['windows'])==3 and result['visual_valid'].sum()==2
    assert set(result['assigned_indices'].tolist())==set(range(50))
    assert len(result['assigned_indices'])==len(torch.unique(result['assigned_indices']))
    assert len(result['encoded_visual_windows'])==2 and result['visual_original_indices']==indices

def test_single_clip_api_scattered_slots_match_pooled_features():
    from src.multimodal.cache import pack_clip_features
    from src.multimodal.fusion import FusionClassifier
    cfg=yaml.safe_load(Path('configs/lab9.yaml').read_text())['model'];m=FusionClassifier(cfg).eval();v={'z':torch.randn(2,512),'indices':[0,2]};a={'z':torch.randn(5,128)};alignment={'visual_valid':torch.tensor([1,0,1],dtype=torch.bool),'audio_valid':torch.ones(5,dtype=torch.bool),'visual_original_indices':[0,2]};mean=torch.randn(640);std=torch.rand(640)+.5;stats={'mean':mean.tolist(),'std':std.tolist()}
    inp=pack_clip_features(v,a,alignment,stats);actual=m(**inp);z=(torch.cat([v['z'].mean(0),a['z'].mean(0)])-mean)/std
    expected=m(visual=z[:512][None],audio=z[512:][None]);assert actual.shape==(1,8) and torch.allclose(actual,expected,atol=1e-6)
    inp['visual'][1]=1e6;assert torch.allclose(m(**inp),actual,atol=1e-6)

def test_clip_api_rejects_cal_before_any_feature_file(tmp_path):
    from src.data.multimodal_data import clip_features
    p=tmp_path/'manifest.json';write_json(p,{'status':'PASS','clips':[{'clip_id':'cal_example','role':'cal','actor_id':8}]})
    with pytest.raises(ValueError,match='does not read cal'):clip_features('cal_example',root=str(tmp_path/'does_not_exist'),manifest_path=p)

@pytest.mark.parametrize('weight',[0.,.5,1.])
def test_registered_late_uses_same_public_interface(weight):
    from src.models.small_cnn import build_model
    from src.multimodal.fusion import late_probabilities
    cfg=yaml.safe_load(Path('configs/lab9.yaml').read_text())['model'];v={**cfg,'strategy':'video'};a={**cfg,'strategy':'audio'};m=build_model({'name':'multimodal_late','video_model':v,'audio_model':a,'weight':weight}).eval();x=torch.randn(4,640)
    for mask in [(1,1),(1,0),(0,1)]:
        args={'visual':x[:,:512],'audio':x[:,512:],'mask':mask};p=m(**args);expected=late_probabilities(m.video(**args),m.audio(**args),weight,mask)
        assert torch.equal(p,expected) and torch.allclose(p.sum(1),torch.ones(4))
    assert all(not p.requires_grad for p in m.parameters())
