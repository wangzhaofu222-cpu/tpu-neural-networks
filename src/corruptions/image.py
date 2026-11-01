import hashlib
import torch
from torchvision.transforms.functional import gaussian_blur, adjust_brightness, adjust_contrast


PARAMETERS={
    'gaussian_noise':[0,.04,.06,.08,.09,.10],
    'gaussian_blur':[0,.4,.7,1.,1.3,1.6],
    'brightness':[1.,1.15,1.3,1.45,1.6,1.75],
    'contrast':[1.,.8,.6,.4,.2,.1],
    'occlusion':[0,.1,.2,.3,.4,.5],
}


def corrupt_image(x,kind,level,seed=1,object_id=''):
    if kind not in PARAMETERS or level not in range(6):raise ValueError('Invalid corruption')
    if level==0:return x
    p=PARAMETERS[kind][level]
    h=int(hashlib.sha256(f'{seed}:{object_id}:{kind}'.encode()).hexdigest()[:15],16)
    g=torch.Generator(device=x.device).manual_seed(h)
    if kind=='gaussian_noise':return (x+p*torch.randn(x.shape,generator=g,device=x.device,dtype=x.dtype)).clamp(0,1)
    if kind=='gaussian_blur':return gaussian_blur(x,[9,9],[p,p])
    if kind=='brightness':return adjust_brightness(x,p)
    if kind=='contrast':return adjust_contrast(x,p)
    # p is occluded area fraction; reproducible location independently per object.
    y=x.clone();H,W=x.shape[-2:];side=max(1,round((p*H*W)**.5));top=int(torch.randint(H-side+1,(),generator=g));left=int(torch.randint(W-side+1,(),generator=g))
    y[...,top:top+side,left:left+side]=.5;return y


def corrupt_record(x,label,object_id,kind,level,seed=1):
    return corrupt_image(x,kind,level,seed,object_id),label,object_id
