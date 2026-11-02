"""Shared-harness integration: resume, macro-F1 selection and role isolation."""
import copy,json
from pathlib import Path
import torch,yaml
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
