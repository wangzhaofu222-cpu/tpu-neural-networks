import numpy as np
import torch
from src.reliability.uncertainty import uncertainty


def selective_risk(U,correct,threshold):
    u=np.asarray(U);c=np.asarray(correct,dtype=bool);accepted=u<=threshold
    return {'coverage':float(accepted.mean()),'risk':float((~c[accepted]).mean()) if accepted.any() else None}


def risk_coverage(U,correct):
    u=np.asarray(U);c=np.asarray(correct,dtype=bool)
    if len(u)!=len(c) or not len(u):raise ValueError('Invalid arrays')
    order=np.argsort(u,kind='stable');risk=np.cumsum(~c[order])/np.arange(1,len(c)+1)
    oracle=np.cumsum(np.sort(~c))/np.arange(1,len(c)+1)
    return {'coverage':(np.arange(1,len(c)+1)/len(c)).tolist(),'risk':risk.tolist(),'AURC':float(risk.mean()),
            'oracle_AURC':float(oracle.mean()),'excess_AURC':float(risk.mean()-oracle.mean()),'tie_rule':'stable object ID order'}


def aurc(U,correct):return risk_coverage(U,correct)['AURC']


class SelectivePredictor:
    def __init__(self, model, measure='msp',temperature=1.):self.model,self.measure,self.temperature=model,measure,temperature
    @torch.no_grad()
    def __call__(self,x,threshold):
        z=self.model(x);return z.argmax(1),uncertainty(z,self.temperature)[self.measure]<=threshold
