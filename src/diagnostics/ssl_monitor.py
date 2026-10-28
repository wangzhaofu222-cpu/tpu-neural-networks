from pathlib import Path
import numpy as np
import torch
from src.ssl.views import Views
from src.evaluation.features import extract_arrays,state_digest
from src.diagnostics.representation import representation_statistics
from src.utils.io import write_json,digest

class RepresentationMonitor:
    def __init__(self,model,train,cfg,path):
        self.model,self.train,self.cfg,self.path=model,train,cfg,Path(path)
        n=cfg['representation']['diagnostic_size'];rng=np.random.default_rng(cfg['profile']['base_seed']+4000)
        import json
        full_train=json.loads(Path(cfg['data']['split_path']).read_text(encoding='utf-8'))['roles']['train']
        self.indices=sorted(rng.choice(full_train,n,replace=False).tolist())
        self.pairing=rng.permutation(n)
        while np.any(self.pairing==np.arange(n)):self.pairing=rng.permutation(n)
        self.views=Views(train.images,self.indices,train.mean,train.std,cfg['augmentations'],cfg['profile']['base_seed']+4001)
        workers=cfg['data']['num_workers'];g=torch.Generator().manual_seed(4002)
        self.loader=torch.utils.data.DataLoader(self.views,batch_size=256,shuffle=False,num_workers=workers,
                    persistent_workers=workers>0,generator=g,pin_memory=cfg['runtime']['device']=='cuda')
        (self.path/'diagnostics').mkdir(exist_ok=True)
        write_json(self.path/'diagnostic_protocol.json',{'image_indices':self.indices,'random_pair_permutation':self.pairing.tolist(),
            'role':'train','augmentation_seed':cfg['profile']['base_seed']+4001,'size':n,'index_fingerprint':digest(self.indices)})
        self.last=None
    @torch.no_grad()
    def observe(self,epoch):
        m=self.model;device=self.cfg['runtime']['device'];before=state_digest(m);was=m.training;m.eval()
        h,z=extract_arrays(m,self.train.images,self.indices,self.train.mean,self.train.std,device)
        simh=[];simz=[]
        for views,_ in self.loader:
            a,b=views[:,0].to(device),views[:,1].to(device);ha,za=m.representations(a);hb,zb=m.representations(b)
            simh.append(torch.nn.functional.cosine_similarity(ha,hb).cpu());simz.append((za*zb).sum(1).cpu())
        positive_h,positive_z=torch.cat(simh),torch.cat(simz)
        random_h=torch.nn.functional.cosine_similarity(h,h[self.pairing]);random_z=(z*z[self.pairing]).sum(1)
        stats={'h':representation_statistics(h),'z_hat':representation_statistics(z,normalized=True)}
        for key,p,r in [('h',positive_h,random_h),('z_hat',positive_z,random_z)]:
            stats[key].update(positive_similarity=p.mean().item(),random_pair_similarity=r.mean().item())
        m.train(was);assert state_digest(m)==before,'Diagnostics changed model/BatchNorm state'
        torch.save({'h':h,'z_hat':z,'positive_similarity_h':positive_h,'positive_similarity_z':positive_z,
                    'random_similarity_h':random_h,'random_similarity_z':random_z,'image_indices':self.indices},self.path/f'diagnostics/epoch_{epoch:03}.pt')
        record={'epoch':epoch,'representation':stats,'RUN_ID':self.path.name}
        write_json(self.path/f'diagnostics/epoch_{epoch:03}.json',record)
        self.last=record;return record
