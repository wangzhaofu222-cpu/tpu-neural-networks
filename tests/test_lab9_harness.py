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
