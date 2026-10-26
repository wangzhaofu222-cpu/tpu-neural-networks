import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import pytest
import torch
import yaml
from src.config import load_config,validate
from src.engine.runner import ExperimentRunner
from src.engine.checkpoint import save_checkpoint,load_checkpoint
from src.models.small_cnn import build_model
from src.metrics.classification import evaluate_loader
from src.data.vision import datasets,loaders
from src.utils.provenance import experiment_id,git_state
from src.utils.io import digest
from src.utils.reproducibility import seed_all,rng_state,restore_rng
from src.profile_gen import generate
from src.analysis.statistics import parameter_relative_difference,seed_statistics,paired_result
from src.compare import diff_configs


def test_L0_T01_config_validation(fixture_config,tmp_path):
    validate(fixture_config)
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(fixture_config),encoding='utf-8')
    assert load_config(path)==fixture_config
    del fixture_config['optimizer']
    with pytest.raises(ValueError):
        validate(fixture_config)


def test_L0_T05_T06_T08_ids_and_code_changes(fixture_config,tmp_path):
    code={'commit':'abc','dirty':False,'source_hash':'a'}
    exp=experiment_id(code,fixture_config,'raw','split')
    assert exp==experiment_id(code,fixture_config,'raw','split')
    changed=copy.deepcopy(fixture_config)
    changed['optimizer']['lr']*=2
    assert exp!=experiment_id(code,changed,'raw','split')
    assert exp!=experiment_id({**code,'source_hash':'b'},fixture_config,'raw','split')
    assert exp!=experiment_id(code,fixture_config,'changed','split')
    (tmp_path/'src').mkdir()
    f=tmp_path/'src/a.py'
    f.write_text('x=1')
    before=git_state(tmp_path)['source_hash']
    f.write_text('x=2')
    assert before!=git_state(tmp_path)['source_hash']


def test_rng_roundtrip_all_libraries():
    seed_all(42,True,1)
    g=torch.Generator().manual_seed(123)
    s=rng_state(g)
    a=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=g))
    restore_rng(s,g)
    b=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=g))
    assert a[:2]==b[:2]
    assert torch.equal(a[2],b[2]) and torch.equal(a[3],b[3])


def test_L0_T03_checkpoint_output_unchanged(fixture_config,tmp_path):
    seed_all(1,True,1)
    m=build_model(fixture_config['model'])
    opt=torch.optim.AdamW(m.parameters())
    sch=torch.optim.lr_scheduler.StepLR(opt,1)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    g=torch.Generator().manual_seed(1)
    x=torch.randn(2,3,32,32)
    before=m(x).detach().clone()
    p=tmp_path/'state.pt'
    save_checkpoint(p,m,opt,sch,scaler,0,0,g)
    with torch.no_grad():
        next(m.parameters()).add_(1)
    load_checkpoint(p,m,opt,sch,scaler,g)
    assert torch.equal(before,m(x))


def test_L0_T04_metrics_deterministic(fixture_config):
    seed_all(1,True,1)
    ds,_=datasets(fixture_config)
    dl=loaders(fixture_config,ds,torch.Generator().manual_seed(1))
    model=build_model(fixture_config['model'])
    assert evaluate_loader(model,dl['val'],'cpu')==evaluate_loader(model,dl['val'],'cpu')


def test_L0_T06_T07_repeat_no_overwrite(fixture_config):
    a=ExperimentRunner(fixture_config).run()
    original={p.name:p.read_bytes() for p in a.iterdir() if p.is_file()}
    b=ExperimentRunner(fixture_config).run()
    ma=json.loads((a/'manifest.json').read_text())
    mb=json.loads((b/'manifest.json').read_text())
    assert ma['EXP_ID']==mb['EXP_ID'] and ma['RUN_ID']!=mb['RUN_ID']
    assert original=={p.name:p.read_bytes() for p in a.iterdir() if p.is_file()}
    for filename in ['metrics.jsonl','steps.jsonl']:
        aa=[json.loads(l) for l in (a/filename).read_text().splitlines()]
        bb=[json.loads(l) for l in (b/filename).read_text().splitlines()]
        for item in aa+bb:
            item.pop('epoch_time',None)
        assert aa==bb


def test_resume_in_new_process_exact_trajectory(fixture_config,tmp_path):
    a=ExperimentRunner(fixture_config).run()
    b=ExperimentRunner(fixture_config).run(stop_after_epoch=2)
    cfg=tmp_path/'resume.yaml'
    cfg.write_text(yaml.safe_dump(fixture_config),encoding='utf-8')
    result=subprocess.run([sys.executable,'-X','utf8','-m','src.train','--config',str(cfg),'--resume',str(b/'checkpoints/last.pt')],capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stderr
    c=Path(result.stdout.strip().splitlines()[-1])
    aa=torch.load(a/'checkpoints/last.pt',weights_only=False)
    cc=torch.load(c/'checkpoints/last.pt',weights_only=False)
    assert parameter_relative_difference(aa['model'],cc['model'])==0
    assert aa['global_step']==cc['global_step']==16
    steps_a=[json.loads(l) for l in (a/'steps.jsonl').read_text().splitlines()]
    steps_c=[json.loads(l) for l in (c/'steps.jsonl').read_text().splitlines()]
    assert steps_a==steps_c


def test_L0_T10_two_profiles_without_code_edit(fixture_config):
    for sid in ['unit-student-alpha','unit-student-beta']:
        profile=generate(sid)
        cfg=copy.deepcopy(fixture_config)
        for section in ['profile','model','optimizer','scheduler','train']:
            cfg[section].update(profile[section])
        cfg['train']['epochs']=1
        cfg['experiment']['seed']=profile['profile']['base_seed']
        path=ExperimentRunner(cfg).run()
        assert json.loads((path/'manifest.json').read_text())['profile']==profile['profile']


def test_L0_T11_cpu(fixture_config):
    fixture_config['train']['epochs']=1
    path=ExperimentRunner(fixture_config).run()
    assert json.loads((path/'manifest.json').read_text())['environment']['device']=='cpu'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_L0_T11_cuda_selected_by_config(fixture_config):
    fixture_config['runtime']['device']='cuda'
    fixture_config['train']['epochs']=1
    path=ExperimentRunner(fixture_config).run()
    assert torch.device(json.loads((path/'manifest.json').read_text())['environment']['device']).type=='cuda'


def test_statistics_and_config_diff():
    result=seed_statistics([1,2,3,4,5])
    assert result['mean']==3 and result['sample_std']==pytest.approx(np.sqrt(2.5))
    assert result['ci95']==pytest.approx([1.0367568385,4.9632431615])
    assert paired_result([2,3,4],[1,2,3])['difference_established']
    assert not paired_result([2,1,4],[1,2,3])['difference_established']
    assert diff_configs({'a':{'b':1}},{'a':{'b':1}})=={}
