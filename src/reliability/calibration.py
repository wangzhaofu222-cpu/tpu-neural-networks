import math
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from src.utils.io import digest


class TemperatureScaler:
    """Val-ID allowlist is authoritative; roles cannot be supplied as a mere label."""
    def __init__(self, val_ids):
        self.val_ids=frozenset(val_ids);self.temperature=1.;self.fit_info=None

    def fit(self, logits, labels, ids, *, role='val', ood=False):
        if role!='val' or ood or not ids or len(set(ids))!=len(ids) or not set(ids)<=self.val_ids:
            raise ValueError('Temperature fit accepts unique val IDs only')
        z=torch.as_tensor(logits,dtype=torch.float64);y=torch.as_tensor(labels,dtype=torch.long)
        if len(z)!=len(ids) or len(y)!=len(ids):raise ValueError('Mismatched IDs')
        def objective(log_t):return torch.nn.functional.cross_entropy(z/math.exp(log_t),y).item()
        # Explicit T=1 candidate guarantees nonincrease even if minimization fails.
        opt=minimize_scalar(objective,bounds=(-8.,8.),method='bounded',options={'xatol':1e-10})
        candidates=[(objective(0.),0.),(objective(opt.x),float(opt.x))]
        loss,lt=min(candidates);self.temperature=math.exp(lt)
        self.fit_info={'temperature':self.temperature,'val_nll_before':objective(0.),'val_nll_after':loss,
                       'method':'bounded scalar minimization of logT [-8,8], plus T=1','optimizer_success':bool(opt.success),
                       'val_IDs':list(ids),'val_IDs_hash':digest(list(ids)),'role':'val','test_used':False}
        return self

    def transform(self,logits):return torch.as_tensor(logits)/self.temperature
