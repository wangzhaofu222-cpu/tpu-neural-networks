import copy
import json
import pytest
import torch
import src.engine.runner as runner
from src.analysis.statistics import parameter_relative_difference


def test_accumulation_weights_unequal_microbatches_by_samples(fixture_config):
    cfg=copy.deepcopy(fixture_config)
    cfg['train'].update(epochs=1,batch_size=40,accumulation_steps=1)
    cfg['optimizer'].update(name='sgd',momentum=0.,weight_decay=0.)
    a=runner.ExperimentRunner(cfg).run()
    cfg['train'].update(batch_size=16,accumulation_steps=3)
    b=runner.ExperimentRunner(cfg).run()
    aa=torch.load(a/'checkpoints/last.pt',weights_only=False)
    bb=torch.load(b/'checkpoints/last.pt',weights_only=False)
    assert aa['global_step']==bb['global_step']==1
    assert parameter_relative_difference(aa['model'],bb['model'])<1e-7


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_amp_overflow_does_not_count_skipped_optimizer_steps(fixture_config,monkeypatch):
    original=runner.build_model
    def overflowing(cfg):
        model=original(cfg)
        for p in model.parameters():
            p.register_hook(lambda grad: torch.full_like(grad,float('inf')))
        return model
    monkeypatch.setattr(runner,'build_model',overflowing)
    fixture_config['runtime']['device']='cuda'
    fixture_config['train'].update(amp=True,epochs=1)
    path=runner.ExperimentRunner(fixture_config).run()
    summary=json.loads((path/'summary.json').read_text())
    steps=[json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
    assert summary['global_step']==0
    assert all(row['optimizer_step_skipped'] for row in steps)
