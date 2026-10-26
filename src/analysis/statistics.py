import numpy as np
from scipy.stats import t


def seed_statistics(values):
    x=np.asarray(values,dtype=np.float64)
    if len(x)<2 or not np.isfinite(x).all():
        raise ValueError('At least two finite values required')
    mean=float(x.mean())
    std=float(x.std(ddof=1))
    radius=float(t.ppf(.975,len(x)-1)*std/np.sqrt(len(x)))
    return {'n':len(x),'mean':mean,'sample_std':std,'ci95':[mean-radius,mean+radius],
            'ci_method':'Student t, df=n-1; independent runs', 'minimum_effect_heuristic':2*std}


def paired_result(a,b):
    if len(a)!=3 or len(b)!=3:
        raise ValueError('Course main comparisons require exactly three paired seeds')
    differences=np.asarray(a)-np.asarray(b)
    stats=seed_statistics(differences)
    stats['differences']=differences.tolist()
    stats['difference_established']=bool((np.all(differences>0) or np.all(differences<0)) and abs(stats['mean'])>=2*stats['sample_std'])
    return stats


def parameter_relative_difference(a,b):
    import torch
    if a.keys()!=b.keys():
        raise ValueError('Parameter names differ')
    aa=torch.cat([v.detach().cpu().double().flatten() for v in a.values()])
    bb=torch.cat([v.detach().cpu().double().flatten() for v in b.values()])
    return ((aa-bb).norm()/(aa.norm()+1e-12)).item()
