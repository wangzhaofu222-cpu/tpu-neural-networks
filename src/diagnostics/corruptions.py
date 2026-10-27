import math
import torch
from torchvision.transforms.functional import gaussian_blur

LEVELS={'gaussian_noise':[.04,.08,.12],'blur':[.5,1.,1.5],'contrast':[.6,.4,.2],
        'occlusion':[6,10,14],'translation':[2,4,8],'patch_shuffle':[16,8,4]}


def corrupt(x,name,level,seed):
    if name not in LEVELS or level not in [0,1,2,3]:raise ValueError('Invalid corruption/level')
    if level==0:return x
    value=LEVELS[name][level-1];g=torch.Generator().manual_seed(seed)
    if name=='gaussian_noise':out=x+torch.randn(x.shape,generator=g,dtype=x.dtype)*value
    elif name=='blur':
        kernel=2*math.ceil(3*value)+1;out=gaussian_blur(x,[kernel,kernel],[value,value])
    elif name=='contrast':out=x.mean()+value*(x-x.mean())
    elif name=='occlusion':
        out=x.clone();h,w=x.shape[-2:];i=torch.randint(h-value+1,(),generator=g).item();j=torch.randint(w-value+1,(),generator=g).item()
        out[:,i:i+value,j:j+value]=.5
    elif name=='translation':
        axis=int(torch.randint(2,(),generator=g));sign=1 if torch.randint(2,(),generator=g) else -1
        out=torch.zeros_like(x);shift=sign*value
        if axis==0:
            if shift>0:out[:,shift:,:]=x[:,:-shift,:]
            else:out[:,:shift,:]=x[:,-shift:,:]
        else:
            if shift>0:out[:,:,shift:]=x[:,:,:-shift]
            else:out[:,:,:shift]=x[:,:,-shift:]
    else:
        c,h,w=x.shape;p=value;gh,gw=h//p,w//p
        blocks=x.reshape(c,gh,p,gw,p).permute(1,3,0,2,4).reshape(gh*gw,c,p,p)
        out=blocks[torch.randperm(gh*gw,generator=g)].reshape(gh,gw,c,p,p).permute(2,0,3,1,4).reshape(c,h,w)
    return out.clamp(0,1)


class CorruptedImages(torch.utils.data.Dataset):
    def __init__(self,base,name,level,seed):self.base,self.name,self.level,self.seed=base,name,level,seed
    def __len__(self):return len(self.base)
    def __getitem__(self,item):
        idx=self.base.indices[item]
        x=torch.from_numpy(self.base.images[idx].copy()).permute(2,0,1).float()/255
        x=corrupt(x,self.name,self.level,(self.seed+idx*1009)%(2**63))
        return (x-self.base.mean)/self.base.std,int(self.base.labels[idx])
