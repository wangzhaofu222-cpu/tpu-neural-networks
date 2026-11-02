import torch


def validate_mask(mask,batch,device):
    m=torch.as_tensor(mask,device=device)
    if m.shape==(2,):m=m.expand(batch,2)
    if m.shape!=(batch,2) or not ((m==0)|(m==1)).all() or (m.sum(1)==0).any():raise ValueError('Only (1,1),(1,0),(0,1) modality masks are valid')
    return m.bool()


def apply_mask(visual,audio,mask):
    m=validate_mask(mask,len(visual),visual.device)
    return torch.where(m[:,:1],visual,torch.zeros_like(visual)),torch.where(m[:,1:],audio,torch.zeros_like(audio))


def sample_modality_mask(batch,p_drop,device='cpu',generator=None):
    if not 0<=p_drop<=1:raise ValueError('Invalid dropout probability')
    u=torch.rand(batch,device=device,generator=generator);m=torch.ones(batch,2,dtype=torch.bool,device=device)
    m[(u>=1-p_drop)&(u<1-p_drop/2),0]=False;m[u>=1-p_drop/2,1]=False
    return m
