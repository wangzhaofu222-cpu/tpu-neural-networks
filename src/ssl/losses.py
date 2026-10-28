import math
import torch
from torch.nn import functional as F

def nt_xent(z1,z2,tau):
    if z1.shape!=z2.shape or len(z1)<2 or tau<=0:raise ValueError('NT-Xent requires matched N>=2 and positive tau')
    # The full2N denominator is retained, even when autocast is enabled for the encoder.
    z=torch.cat((z1,z2),0)
    if z.dtype in [torch.float16,torch.bfloat16]:z=z.float()
    z=F.normalize(z,dim=1);n=len(z1);size=2*n
    with torch.autocast(device_type=z.device.type,enabled=False):
        s=z@z.T;positive=(torch.arange(size,device=z.device)+n)%size
        diagonal=torch.eye(size,device=z.device,dtype=torch.bool)
        scores=(s/tau).masked_fill(diagonal,float('-inf'))
        losses=torch.logsumexp(scores,1)-scores[torch.arange(size,device=z.device),positive]
        negmask=~diagonal;negmask[torch.arange(size,device=z.device),positive]=False
        stats={'positive_similarity':s[torch.arange(size,device=z.device),positive].mean().detach().item(),
               'negative_similarity':s[negmask].mean().detach().item(),'N':n,'negative_count':size-2,'random_baseline_loss':math.log(size-1)}
    return losses.mean(),stats
