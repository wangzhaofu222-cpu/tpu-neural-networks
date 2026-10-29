import hashlib,json,importlib.metadata
from pathlib import Path
import torch,yaml
from src.profile_gen import generate
from src.generative.distributions import distribution,make_roles
from src.utils.io import write_json,file_hash

def main():
    out=Path('experiments/lab6');out.mkdir(exist_ok=True)
    sid=json.loads(Path('.private/lab4_student_id.json').read_text(encoding='utf-8'))['student_id']
    p=generate(sid);profile=p['profile'];profile.update({k:p[k] for k in ['lab6_target','lab6_eta','lab6_scale']})
    write_json(out/'profile.json',{'profile':profile,'lab6_hash':hashlib.sha256((sid+':lab6').encode()).hexdigest()})
    pdf=Path('drive-download-20261026T053112Z-1-001/lab6_ebm_normalizing_flows_1010.pdf')
    write_json(out/'teacher_source.json',{'file':str(pdf),'sha256':file_hash(pdf),'pages':15,'revision':4})
    lock=Path('docs/environment/lab6-requirements.lock.txt')
    lock.write_text('\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions()))+'\n',encoding='utf-8')
    cfg={'experiment':{'name':'lab6','lab':6,'seed':1,'deterministic':True,'purpose':'development','role':'E1'},
        'data':{'name':'gaussian_mixture_2d','target':'two_blobs','train_fraction':1.,'num_workers':0},
        'model':{'name':'energy_model','width':128,'depth':3,'cap':profile['lab6_scale']},
        'optimizer':{'name':'adam','lr':.001,'weight_decay':0.,'momentum':0.},
        'scheduler':{'name':'constant','unit':'epoch','warmup_epochs':0},
        'train':{'epochs':65,'max_steps':5000,'batch_size':256,'amp':False,'accumulation_steps':1},
        'sampling':{'eta':profile['lab6_eta'][1],'temperature':1.,'steps':50,'boundary':'reflection'},
        'generative':{'energy_regularization':0.,'checkpoint_selection':'fixed5000step_endpoint'},
        'logging':{'run_root':'runs/lab6','save_every':1,'save_every_steps':500,'console_every_steps':500,'keep_epochs':[16,32,48]},
        'runtime':{'device':'cpu','threads':2,'max_seconds':0,'dependency_lock':str(lock).replace('\\','/'),'pause_file':'experiments/lab6/PAUSE'},
        'profile':profile}
    Path('configs/lab6.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    metadata={}
    for name in ['two_blobs','eight_gaussians','four_corners','three_blobs']:
        roles=make_roles(name);dest=Path('data/lab6')/name;dest.mkdir(parents=True,exist_ok=True)
        for r,x in roles.items():torch.save(x,dest/f'{r}.pt')
        target=distribution(name)
        metadata[name]={'centers':target.centers.tolist(),'sigma':target.sigma,'role_counts':{r:len(x) for r,x in roles.items()},
            'role_seeds':{'train':601,'val':602,'test':603},'cal_created':False,'standardization':False,'domain':target.domain,
            'role_tensor_sha256':{r:hashlib.sha256(x.numpy().tobytes()).hexdigest() for r,x in roles.items()},
            'exact_mode_probabilities_outlier_first':target.mode_probabilities().tolist()}
    write_json(out/'data_protocol.json',metadata)
    print({'profile':profile,'two_blobs_logp_origin':float(distribution('two_blobs').log_prob(torch.zeros(1,2,dtype=torch.float64)))})
if __name__=='__main__':main()
