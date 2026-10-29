"""Required rare-target retraining and counterfactual10T/zero-predictor controls."""
import copy,math
from pathlib import Path
import torch
from src.diffusion.schedules import make_schedule,terminal_kl
from src.data.gmm2d import roles_and_stats,standardize,target
from src.utils.io import write_json
from reports.tools.lab7_common import R,base,read,registry,train,evaluate,metric_protocol


def main():
    torch.set_num_threads(2)
    if (R/'supplements.json').exists():return
    c=base();lock=read(R/'E3_protocol_lock.json');A=lock['A'];rare=[]
    for multiplier in [.5,.25]:
        cfg=copy.deepcopy(c);cfg['data']['rare_weight_multiplier']=multiplier;cfg['data']['pi0']=c['data']['pi0']*multiplier
        cfg['diffusion']['kind']=A['schedule'];cfg['diffusion']['alpha_bar_T']=float(make_schedule(cfg['diffusion']).alpha_bar[-1])
        cfg['experiment']['seed']=1
        source=train('rare_'+str(multiplier),cfg)
        q=evaluate('rare_'+str(multiplier)+'_A',source,A['sampler'],A['K'])
        _,labels=target(cfg).sample(20000,cfg['data']['seeds']['train'],True)
        roles,stats=roles_and_stats(cfg)
        rare.append({'multiplier':multiplier,'pi0':cfg['data']['pi0'],'train_component0_count':int((labels==0).sum()),
            'train_geometric_mode0_count':int((target(cfg).mode_id(roles['train'])==0).sum()),
            'training_source':str(source),'normalization':stats,'quality':q})
    cfg=copy.deepcopy(c);cfg['diffusion']['T']*=10;cfg['diffusion']['alpha_bar_T']=float(make_schedule(cfg['diffusion']).alpha_bar[-1])
    source=train('Q1_T_times10',cfg)
    q=evaluate('Q1_T_times10_DDPM',source,timing_repeats=5,warmup=True)
    roles,stats=roles_and_stats(cfg);s=make_schedule(cfg['diffusion'])
    q1={'training_source':str(source),'quality':q,'alpha_bar_T':float(s.alpha_bar[-1]),
        'terminal_KL':terminal_kl(s,standardize(roles['train'],stats))}
    original=registry()['training']['E1_linear']
    zero=evaluate('Q2_zero_predictor_DDPM',original,predictor='zero')
    sample=torch.load(Path('runs/lab7')/zero['RUN_ID']/'standardized_samples.pt',weights_only=False).double()
    s=make_schedule(c['diffusion']);variance=1.
    for t in range(s.T,0,-1):
        variance=variance/float(s.alpha[t])+(float(s.posterior_var[t]) if t>1 else 0.)
    q2={'quality':zero,'theory_variance_standardized':variance,'measured_variance_standardized':sample.var(0).tolist(),
        'relative_variance_error':(sample.var(0)/variance-1).abs().tolist(),
        'derivation':'v_T=1; v_(t-1)=v_t/alpha_t+beta_tilde_t for t>1; v_0=v_1/alpha_1, no final added noise'}
    write_json(R/'supplements.json',{'rare_boundaries':rare,'Q1':q1,'Q2':q2})
    print('SUPPLEMENTS COMPLETE',flush=True)


if __name__=='__main__':main()
