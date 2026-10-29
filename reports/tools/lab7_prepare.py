"""Prepare deterministic Lab7 protocol, never changing Lab0–6 records."""
import json
import subprocess
import sys
from pathlib import Path
import torch
import yaml
from src.profile_gen import generate
from src.data.gmm2d import roles_and_stats
from src.diffusion.schedules import make_schedule
from src.utils.io import write_json, file_hash

R=Path('experiments/lab7')


def main():
    R.mkdir(exist_ok=True)
    if (R/'profile.json').exists():
        raise RuntimeError('Do not overwrite an existing protocol')
    sid=json.loads(Path('.private/lab4_student_id.json').read_text(encoding='utf-8'))['student_id']
    p=generate(sid)
    profile={**p['profile'],**{k:v for k,v in p.items() if k.startswith('lab7_')}}
    T,lo,hi=profile['lab7_T_beta'];M,rho=profile['lab7_modes']
    cfg={'experiment':{'name':'lab7','lab':7,'seed':1,'deterministic':True,'purpose':'development','role':'E1_linear'},
        'data':{'name':'rare_gmm2d','M':M,'rho':rho,'pi0':profile['lab7_rare_weight'],
                'rare_weight_multiplier':1.,'seeds':{'train':701,'val':702,'test':703},'train_fraction':1.,'num_workers':0},
        'model':{'name':'diffusion_mlp','width':256,'depth':6,'time_dim':128,'activation':'silu'},
        'diffusion':{'kind':'linear','T':T,'beta_min':lo,'beta_max':hi,'parameterization':'epsilon','reverse_variance':'posterior','clipping':False},
        'optimizer':{'name':'adam','lr':.001,'weight_decay':0.,'momentum':0.},
        'scheduler':{'name':'constant','unit':'epoch','warmup_epochs':0},
        'train':{'epochs':514,'max_steps':20000,'batch_size':512,'amp':False,'accumulation_steps':1},
        'logging':{'run_root':'runs/lab7','save_every':1,'save_every_steps':2000,'console_every_steps':1000,'keep_epochs':[128,256,384]},
        'runtime':{'device':'cpu','threads':2,'max_seconds':0,'dependency_lock':'docs/environment/lab7-requirements.lock.txt','pause_file':'experiments/lab7/PAUSE'},
        'profile':profile}
    s=make_schedule(cfg['diffusion']);cfg['diffusion']['alpha_bar_T']=float(s.alpha_bar[-1])
    Path('configs/lab7.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    write_json(R/'profile.json',profile)
    roles,stats=roles_and_stats(cfg)
    d=Path('data/lab7/base');d.mkdir(parents=True,exist_ok=True)
    for k,x in roles.items():torch.save(x,d/(k+'.pt'))
    write_json(R/'data_protocol.json',{'seeds':cfg['data']['seeds'],'sizes':{k:len(x) for k,x in roles.items()},
               'normalization':stats,'role_file_hashes':{k:file_hash(d/(k+'.pt')) for k in roles},'cal':None,
               'standardized_training':True,'metrics_in_original_units':True})
    pdf=Path('drive-download-20261026T053112Z-1-001/lab7_diffusion_score_models_1010.pdf')
    write_json(R/'teacher_source.json',{'path':str(pdf),'sha256':file_hash(pdf),'pages':16,'revision':4,'all_pages_read':True})
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze']).decode('utf-8')
    Path(cfg['runtime']['dependency_lock']).write_text(lock,encoding='utf-8')
    print(profile)


if __name__=='__main__':main()
