"""Synthetic tasks from Lab3 §6; no padding and one length per batch."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from src.utils.io import digest,write_json

BLANK,KEY,VALUE,QUERY,MARKER=64,65,66,67,68
ROLES={'train':1009,'val':2003,'cal':3001,'test':4001}

def example(task,length,rng):
    if task=='copy': length-=length%2
    if length<12: raise ValueError('Sequence length must be >=12')
    x=rng.integers(0,64,length,dtype=np.int64);y=np.full(length,-100,dtype=np.int64)
    target=np.full(length,-1,dtype=np.int64)
    if task=='copy':
        half=length//2;x[half:]=BLANK;y[half:]=x[:half];target[half:]=np.arange(half)
    elif task=='needle':
        j=int(rng.integers(0,length-2));x[j]=MARKER;x[-1]=QUERY;y[-1]=x[j+1];target[-1]=j+1
    elif task=='retrieval':
        n=min(32,max(2,length//8));keys=rng.choice(64,n,replace=False);values=rng.integers(0,64,n)
        # Uniform random gap allocation permits every pair to occupy varying locations.
        gaps=rng.multinomial(length-2-4*n,np.full(n+1,1/(n+1)))
        j=int(rng.integers(n));pos=int(gaps[0]);positions=[]
        for i in range(n):
            x[pos:pos+4]=[KEY,keys[i],VALUE,values[i]];positions.append(pos+3);pos+=4+int(gaps[i+1])
        x[-2:]=[QUERY,keys[j]];y[-1]=values[j];target[-1]=positions[j]
    else: raise ValueError(task)
    return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(target)

class Sequences(Dataset):
    def __init__(self,task,L0,seed,role,size,batch_size=32,length=None):
        if role not in ROLES:raise ValueError(role)
        self.task,self.L0,self.seed,self.role,self.size,self.batch_size,self.length=task,L0,seed,role,size,batch_size,length
        self.epoch=0
    def __len__(self):return self.size
    def set_epoch(self,epoch):self.epoch=epoch
    def sample(self,i):
        if self.role=='cal':raise PermissionError('cal reserved until Lab8')
        return self._generate(i)
    def _generate(self,i):
        epoch=self.epoch if self.role=='train' else 0
        length=self.length
        if length is None:
            g=np.random.default_rng(np.random.SeedSequence([self.seed,ROLES[self.role],epoch,i//self.batch_size,0]))
            length=int(g.integers(self.L0//2,self.L0+1))
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,ROLES[self.role],epoch,i,1,length]))
        return example(self.task,length,rng)
    def __getitem__(self,i):return self.sample(i)[:2]

def sequence_datasets(cfg):
    d=cfg['data'];seed=cfg['experiment']['seed'];b=cfg['train']['batch_size'];L0=d['L0']
    spec={'version':1,'task':d['task'],'L0':L0,'seed':seed,'role_offsets':ROLES,'vocabulary':64,
          'train_length':'uniform integer L0/2..L0 per homogeneous batch; copy odd -> L-1',
          'steps_per_epoch':d['steps_per_epoch'],'batch_size':b,'evaluation_examples':d['evaluation_examples']}
    ds={'train':Sequences(d['task'],L0,seed,'train',b*d['steps_per_epoch'],b),
        'val':Sequences(d['task'],L0,seed,'val',d['evaluation_examples'],b,L0)}
    # Materialize cal exactly once. It is never opened by training/evaluation.
    calpath=Path(d['root'])/('cal-'+digest(spec)[:20]+'.json')
    if not calpath.exists():
        cal=Sequences(d['task'],L0,seed,'cal',d['evaluation_examples'],b,L0)
        write_json(calpath,{'role':'cal','reserved_until':8,'specification':spec,
                           'examples':[[v.tolist() for v in cal._generate(i)] for i in range(len(cal))]})
    return ds,{'dataset':'Lab3 synthetic generator v1','data_fingerprint':digest(spec),
        'split_fingerprint':digest({'role_seeds':{k:seed+v for k,v in ROLES.items()},'generator':spec}),
        'split_sizes':{'train':len(ds['train']),'val':len(ds['val']),'cal':d['evaluation_examples'],'test_per_length':d['evaluation_examples']},
        'generator_manifest':spec,'cal_path':calpath.as_posix(),'synthetic_fixture':False,'actual_train_size':len(ds['train'])}

def sequence_loaders(cfg,ds,generator):
    return {role:DataLoader(v,batch_size=cfg['train']['batch_size'],shuffle=False,num_workers=0)
            for role,v in ds.items()}
