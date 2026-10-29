import math
import torch

def bandwidth(train,seed=604):
    idx=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed))[:2000]
    return float(torch.pdist(train[idx].double()).median()),idx

def kernel_mean(x,y,h,chunk=512):
    x=x.detach().cpu().double();y=y.detach().cpu().double();total=0.;yy=y.square().sum(-1)[None,:]
    for block in x.split(chunk):
        d=(block.square().sum(-1)[:,None]+yy-2*block@y.T).clamp_min_(0)
        total+=torch.exp(-d/(2*h*h)).sum().item()
    return total/(len(x)*len(y))

def mmd_squared(x,y,h,reference_self=None):
    xx=kernel_mean(x,x,h) if reference_self is None else reference_self
    value=xx+kernel_mean(y,y,h)-2*kernel_mean(x,y,h)
    return max(0.,value)  # only numerical roundoff; empirical V-statistic is nonnegative

def mode_frequencies(samples,target):
    ids=target.mode_id(samples);counts=torch.bincount(ids+1,minlength=len(target.centers)+1)
    return counts.double()/len(samples),counts

def sample_metrics(samples,reference,target,h,reference_self=None):
    empirical,counts=mode_frequencies(samples,target);ref,refcounts=mode_frequencies(reference,target)
    exact=target.mode_probabilities()
    return {'Cov':float((empirical[1:]>=exact[1:]/2).double().mean()),
        'Dmode':float((empirical-ref).abs().sum()/2),'Dmode_exact_target':float((empirical-exact).abs().sum()/2),
        'MMD2':mmd_squared(reference,samples,h,reference_self),'mode_counts_including_outlier_first':counts.tolist(),
        'reference_mode_counts_including_outlier_first':refcounts.tolist(),'exact_mode_probabilities_outlier_first':exact.tolist(),
        'MMD_estimator':'empirical V-statistic, including diagonals','Dmode_estimator':'two-sample plug-in using fixed5000test reference including outliers'}

def paired_statistics(a,b,noise):
    delta=torch.tensor(a,dtype=torch.float64)-torch.tensor(b,dtype=torch.float64)
    if len(delta)!=3:raise ValueError('Three paired seeds required')
    mean=float(delta.mean());sd=float(delta.std(unbiased=True));same=bool((delta>0).all() or (delta<0).all())
    established=same and abs(mean)>=2*sd and abs(mean)>=noise
    return {'deltas':delta.tolist(),'mean':mean,'sample_sd':sd,'same_nonzero_sign':same,'abs_mean_ge_2sd':abs(mean)>=2*sd,
        'metric_noise':noise,'abs_mean_ge_noise':abs(mean)>=noise,'conclusion':'различие установлено' if established else 'различие не установлено'}

def select_flow_depth(rows,pointwise_nll):
    ordered=sorted(rows,key=lambda r:r['depth']);comparisons=[];chosen=None
    for a,b in zip(ordered,ordered[1:]):
        diff=pointwise_nll[a['depth']].double()-pointwise_nll[b['depth']].double()
        gain=float(diff.mean());sem=float(diff.std(unbiased=True)/math.sqrt(len(diff)));clear=gain>2*sem
        comparisons.append({'from':a['depth'],'to':b['depth'],'val_NLL_improvement':gain,'paired_difference_SEM':sem,'clear':clear})
        if chosen is None and not clear:chosen=a['depth']
    return {'selected_depth':8 if chosen is None else chosen,'all_improvements_clear':chosen is None,'adjacent_comparisons':comparisons,
        'rule':'smallest depth with next paired val NLL gain<=2SEM; if every gain clear use8; test excluded'}
