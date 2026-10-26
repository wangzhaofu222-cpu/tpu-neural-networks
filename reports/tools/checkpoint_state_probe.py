"""One-gradient controlled diagnostic, not a second training framework."""
import argparse
import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from src.config import load_config
from src.data.vision import datasets,loaders
from src.models.small_cnn import build_model
from src.utils.io import digest,file_hash,write_json
from src.utils.provenance import create_run,git_state
from src.utils.reproducibility import seed_all,restore_rng
from src.analysis.statistics import parameter_relative_difference


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    cfg=load_config(a.parent_run/'config.resolved.yaml')
    state=torch.load(a.parent_run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    cfg['experiment'].update(purpose='diagnostic',name='lab0_CF03_component_probe',diagnostic_script_sha256=file_hash(__file__))
    cfg['runtime']['device']='cpu'
    seed_all(cfg['experiment']['seed'],True,cfg['runtime']['threads'])
    ds,info=datasets(cfg)
    saved_g=torch.Generator()
    saved_g.set_state(state['loader_rng'])
    resumed_batch=next(iter(loaders(cfg,ds,saved_g)['train']))
    fresh_g=torch.Generator().manual_seed(cfg['experiment']['seed'])
    fresh_batch=next(iter(loaders(cfg,ds,fresh_g)['train']))
    batch_hash=lambda pair:digest({'x':pair[0].tolist(),'y':pair[1].tolist()})
    model=build_model(cfg['model'])
    model.load_state_dict(state['model'])
    model.train()
    loss=F.cross_entropy(model(resumed_batch[0]),resumed_batch[1])
    loss.backward()
    gradients={k:p.grad.detach().clone() for k,p in model.named_parameters()}
    lr=state['optimizer']['param_groups'][0]['lr']
    def update(load_moments,learning_rate):
        m=build_model(cfg['model']); m.load_state_dict(state['model'])
        ocfg=cfg['optimizer']
        if ocfg['name']!='adamw':
            raise ValueError('This diagnostic is explicitly written for the assigned AdamW profile')
        opt=torch.optim.AdamW(m.parameters(),lr=learning_rate,weight_decay=ocfg['weight_decay'])
        if load_moments: opt.load_state_dict(copy.deepcopy(state['optimizer']))
        for group in opt.param_groups: group['lr']=learning_rate
        for name,p in m.named_parameters(): p.grad=gradients[name].clone()
        opt.step()
        return {k:v.detach().clone() for k,v in m.state_dict().items()}
    full=update(True,lr)
    reset_moments=update(False,lr)
    reset_lr=update(True,cfg['optimizer']['lr'])
    checkpoint_sha=file_hash(a.parent_run/'checkpoints/last.pt')
    info['data_fingerprint']=digest({'data':info['data_fingerprint'],'checkpoint':checkpoint_sha})
    path,manifest=create_run(cfg,git_state(ROOT),info,model,parent_run_id=a.parent_run.name)
    result={'RUN_ID':path.name,'parent_RUN_ID':a.parent_run.name,'checkpoint_sha256':checkpoint_sha,
        'fixed_gradient_loss':loss.item(),'same_gradient_for_all_updates':True,'resumed_lr':lr,'initial_lr':cfg['optimizer']['lr'],
        'delta_theta_reset_optimizer_moments_only':parameter_relative_difference(full,reset_moments),
        'delta_theta_reset_learning_rate_only':parameter_relative_difference(full,reset_lr),
        'saved_loader_next_batch_hash':batch_hash(resumed_batch),'fresh_loader_next_batch_hash':batch_hash(fresh_batch),
        'data_order_changes_without_loader_rng':batch_hash(resumed_batch)!=batch_hash(fresh_batch),
        'python_numpy_global_rng_not_used_by_this_deterministic_CNN_pipeline':True,
        'status':'ONE_STEP_DIAGNOSTIC_NOT_A_FULL_TRAINING_RUN'}
    write_json(path/'summary.json',result)
    manifest['status']='COMPLETED_DIAGNOSTIC'
    write_json(path/'manifest.json',manifest)
    write_json(a.out,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
