import numpy as np
import torch
import torch.nn.functional as F


def ece(confidence, correct, M=15, binning='equal_width'):
    c=np.asarray(confidence,dtype=float); a=np.asarray(correct,dtype=float)
    if len(c)==0 or c.shape!=a.shape or not np.isfinite(c).all() or np.any((c<0)|(c>1)) or M<1:
        raise ValueError('Invalid confidence/correctness')
    if binning=='equal_width':
        ids=np.minimum((c*M).astype(int),M-1)
    elif binning=='equal_count':
        ids=np.empty(len(c),dtype=int);ids[np.argsort(c,kind='stable')]=np.minimum(np.arange(len(c))*M//len(c),M-1)
    else: raise ValueError('Unknown binning')
    bins=[];value=0.
    for i in range(M):
        mask=ids==i;n=int(mask.sum());acc=float(a[mask].mean()) if n else None;conf=float(c[mask].mean()) if n else None
        bins.append({'bin':i,'count':n,'accuracy':acc,'confidence':conf})
        if n:value+=n/len(c)*abs(acc-conf)
    return {'value':float(value),'M':M,'binning':binning,'empty_bins':sum(b['count']==0 for b in bins),'bins':bins}


def classification_metrics(logits, labels, *, ood=False, temperature=1., M=15, binning='equal_width'):
    if ood: raise ValueError('No true class: accuracy, ECE, NLL and Brier undefined for OOD')
    if temperature<=0 or not np.isfinite(temperature):raise ValueError('Positive finite temperature required')
    z=torch.as_tensor(logits,dtype=torch.float64)/temperature;y=torch.as_tensor(labels,dtype=torch.long)
    lp=F.log_softmax(z,dim=1);p=lp.exp();pred=z.argmax(1);correct=pred.eq(y);conf=p.max(1).values
    return {'accuracy':correct.double().mean().item(),'mean_confidence':conf.mean().item(),
            'ece':ece(conf.cpu().numpy(),correct.cpu().numpy(),M,binning),
            'nll':F.nll_loss(lp,y).item(),'brier':((p-F.one_hot(y,z.shape[1]))**2).sum(1).mean().item(),'N':len(y),'temperature':float(temperature)}
