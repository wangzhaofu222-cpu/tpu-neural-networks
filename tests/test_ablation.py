import copy
import torch
from src.engine.runner import ExperimentRunner
from src.analysis.statistics import parameter_relative_difference


def test_true_model_only_resume_is_distinct_from_full_resume(fixture_config):
    a=ExperimentRunner(fixture_config).run()
    b=ExperimentRunner(fixture_config).run(stop_after_epoch=2)
    state=torch.load(b/'checkpoints/last.pt',weights_only=False)
    stripped=b/'checkpoints/model_only.pt'
    torch.save({'model':state['model']},stripped)
    config=copy.deepcopy(fixture_config)
    config['experiment']['resume_ablation']='weights_only'
    c=ExperimentRunner(config).run(resume=stripped,weights_only_epoch=2)
    cc=torch.load(c/'checkpoints/last.pt',weights_only=False)
    aa=torch.load(a/'checkpoints/last.pt',weights_only=False)
    assert parameter_relative_difference(aa['model'],cc['model'])>0
    assert aa['global_step']==cc['global_step']
