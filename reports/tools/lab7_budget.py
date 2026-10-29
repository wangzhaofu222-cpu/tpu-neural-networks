import copy,json,time
from pathlib import Path
import torch,yaml
from src.engine.runner import ExperimentRunner
from src.models.small_cnn import build_model
from src.diffusion.schedules import make_schedule
from src.diffusion.samplers import ddpm_sample
from src.data.gmm2d import target,roles_and_stats,ExactPredictor
from src.utils.io import write_json


def main():
    cfg=yaml.safe_load(Path('configs/lab7.yaml').read_text(encoding='utf-8'));rows=[]
    for device in ['cpu','cuda']:
        if device=='cuda' and not torch.cuda.is_available():continue
        c=copy.deepcopy(cfg);c['runtime']['device']=device;c['experiment']['purpose']='budget'
        c['experiment']['role']='budget_'+device;c['train']['max_steps']=120;c['logging']['keep_epochs']=[]
        c['runtime']['max_seconds']=180
        p=ExperimentRunner(c).run();summary=json.loads((p/'summary.json').read_text())
        model=build_model(c['model']).to(device);model.load_state_dict(torch.load(p/'checkpoints/last.pt',weights_only=False)['model']);model.eval()
        short=make_schedule({**c['diffusion'],'T':100},diagnostic=True).to(device)
        x=torch.randn(1000,2,device=device)
        ddpm_sample(model,short,x[:64])
        _,info=ddpm_sample(model,short,x)
        rows.append({'device':device,'width':256,'RUN_ID':p.name,'training_steps':120,
            'training_seconds':summary['training_seconds'],'projected_20000_seconds':summary['training_seconds']/120*20000,
            'sampling_100steps_1000points_seconds':info['seconds'],'projected_T1000_5000points_seconds':info['seconds']*50})
        print('BUDGET',json.dumps(rows[-1]),flush=True)
    chosen=min(rows,key=lambda r:r['projected_20000_seconds']+r['projected_T1000_5000points_seconds']*20)
    cfg['runtime']['device']=chosen['device']
    s=make_schedule(cfg['diffusion']).to(chosen['device']);x=torch.randn(5000,2,device=chosen['device'])
    model=build_model(cfg['model']).to(chosen['device']).eval()
    _,full=ddpm_sample(model,s,x)
    _,stats=roles_and_stats(cfg);exact=ExactPredictor(target(cfg),stats,s)
    _,exact_info=ddpm_sample(exact,s,x.double())
    result={'measurements':rows,'selected_device':chosen['device'],'selected_width':256,
        'full_DDPM_5000_network_seconds':full['seconds'],'full_DDPM_5000_exact_seconds':exact_info['seconds'],
        'reason':'retain default256 width; choose measured faster stable device; no training budget reduction',
        'estimated_9_trainings_seconds':chosen['projected_20000_seconds']*9,
        'estimated_E2_7conditions_6passes_network_seconds':full['seconds']*6*2.3,
        'estimated_E2_7conditions_6passes_exact_seconds':exact_info['seconds']*6*2.3}
    write_json('experiments/lab7/budget.json',result)
    Path('configs/lab7.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
