import hashlib
from pathlib import Path
import torch
from src.utils.io import digest
from src.data.ssl_data import bundle,role_indices,labelled_targets
from src.data.vision import ArrayImages

def state_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def extract_arrays(model,images,indices,mean,std,device,batch=256):
    was_training=model.training;model.eval();hs=[];zs=[]
    try:
        for lo in range(0,len(indices),batch):
            raw=torch.from_numpy(images[indices[lo:lo+batch]].copy()).permute(0,3,1,2).float().to(device)/255
            x=(raw-torch.tensor(mean,device=device)[None,:,None,None])/torch.tensor(std,device=device)[None,:,None,None]
            h,z=model.representations(x);hs.append(h.cpu());zs.append(z.cpu())
    finally:model.train(was_training)
    return torch.cat(hs),torch.cat(zs)

def extract_role(model,cfg,role,device):
    if role=='cal':raise PermissionError('cal reserved untilLab8')
    images,split,mean,std,_=bundle(cfg['data']['root'],cfg['data']['split_path'])
    ids=role_indices(split,role)
    if role=='test':
        from torchvision.datasets import CIFAR10
        test=CIFAR10(cfg['data']['root'],train=False,download=False);images=test.data;labels=torch.tensor(test.targets)
    else:labels=torch.tensor(labelled_targets(cfg['data']['root']))
    before=state_digest(model);h,z=extract_arrays(model,images,ids,mean,std,device)
    assert state_digest(model)==before,'Feature extraction changed encoder state'
    return {'h':h,'z_hat':z,'labels':labels[ids],'indices':ids,'role':role,'encoder_state_sha256':before}
