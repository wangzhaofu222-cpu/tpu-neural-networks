import hashlib
import torch
import torch.nn.functional as F
from torchvision import transforms


POLICIES={'A0','A1','randaugment+cutmix','trivialaugment+mixup'}


def augment_image(x,policy,seed):
    if policy not in POLICIES: raise ValueError('Unknown augmentation policy')
    if policy=='A0':return x
    g=torch.Generator().manual_seed(seed)
    padded=F.pad(x,(4,4,4,4));i,j=torch.randint(9,(2,),generator=g).tolist()
    x=padded[:,i:i+32,j:j+32]
    if torch.rand((),generator=g)<.5:x=x.flip(-1)
    if policy not in {'A0','A1'}:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            op=transforms.RandAugment(num_ops=2,magnitude=9) if policy=='randaugment+cutmix' else transforms.TrivialAugmentWide()
            x=op((x*255).round().to(torch.uint8)).float()/255
    return x


def mix_batch(x,y,policy,classes=10):
    if policy in {'A0','A1'}:return x,y,{}
    alpha=1.
    lam=float(torch.distributions.Beta(alpha,alpha).sample())
    order=torch.randperm(len(y),device='cpu').to(x.device)
    targets=F.one_hot(y,classes).to(dtype=x.dtype)
    if policy=='trivialaugment+mixup':
        return x*lam+x[order]*(1-lam),targets*lam+targets[order]*(1-lam),{'lambda':lam,'kind':'mixup'}
    if policy!='randaugment+cutmix':raise ValueError('Unknown batch augmentation')
    h,w=x.shape[-2:];side=(1-lam)**.5
    cy,cx=torch.randint(h,(1,)).item(),torch.randint(w,(1,)).item()
    hh,ww=int(h*side),int(w*side)
    y0,y1=max(0,cy-hh//2),min(h,cy+hh//2)
    x0,x1=max(0,cx-ww//2),min(w,cx+ww//2)
    out=x.clone();out[:,:,y0:y1,x0:x1]=x[order,:,y0:y1,x0:x1]
    actual=1-((y1-y0)*(x1-x0))/(h*w)
    return out,targets*actual+targets[order]*(1-actual),{'lambda':actual,'kind':'cutmix','rectangle':[y0,y1,x0,x1]}


class AugmentedImages(torch.utils.data.Dataset):
    def __init__(self,base,policy,seed):
        self.base,self.policy,self.seed=base,policy,seed;self.epoch=0
    def __len__(self):return len(self.base)
    def set_epoch(self,epoch):self.epoch=epoch
    def __getitem__(self,item):
        idx=self.base.indices[item]
        x=torch.from_numpy(self.base.images[idx].copy()).permute(2,0,1).float()/255
        seed=int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{idx}'.encode()).digest()[:8],'big')%(2**63)
        x=augment_image(x,self.policy,seed)
        return (x-self.base.mean)/self.base.std,int(self.base.labels[idx])
