"""Full finite quadratic chains, accelerated exact AR1 recurrence; not analytic draws."""
import math
import numpy as np
import torch
from scipy.signal import lfilter

def quadratic_chains(eta,temperature=1.,seed=1,chains=8,burn_in=10000,draws=200000,control_reference_eta=None):
    sigma=math.sqrt(2*(eta if control_reference_eta is None else control_reference_eta)*temperature)
    ss=np.random.SeedSequence(seed);noise=np.stack([np.random.default_rng(s).standard_normal(burn_in+draws) for s in ss.spawn(chains)])
    values=lfilter([sigma],[1.,-(1-eta)],noise,axis=1)
    traj=np.concatenate([np.zeros((chains,1)),values],axis=1)
    kept=values[:,burn_in:];var=kept.var(axis=1,ddof=1);measured=float(var.mean());theory=sigma*sigma/(2*eta-eta*eta)
    return ({'eta':eta,'sigma':sigma,'effective_temperature':sigma*sigma/(2*eta),'chains':chains,'burn_in':burn_in,'draws_per_chain':draws,
        'seed':seed,'variance_per_chain':var.tolist(),'measured_variance':measured,'theory_variance':theory,'relative_error':abs(measured-theory)/theory,
        'acceptance':'PASS' if abs(measured-theory)/theory<=.05 else 'FAIL',
        'method':'scipy lfilter executes all finite AR1 steps; eight independent innovations; no stationary analytic sampling; double precision'},
        torch.from_numpy(traj),torch.from_numpy(noise))

def mode_switches(trajectory,target):
    modes=target.mode_id(trajectory.detach());last=modes[0].clone();counts=torch.zeros_like(last)
    for cur in modes[1:]:
        counts+=((last>=0)&(cur>=0)&(last!=cur)).long()
        last=torch.where(cur>=0,cur,last)
    return {'mode_switch_count':int(counts.sum()),'chain_ever_changed_valid_mode_fraction':float((counts>0).float().mean()),
        'mode_label_changes_including_outliers':int((modes[1:]!=modes[:-1]).sum()),
        'final_other_mode_fraction':float(((modes[0]>=0)&(modes[-1]>=0)&(modes[-1]!=modes[0])).float().mean())}
