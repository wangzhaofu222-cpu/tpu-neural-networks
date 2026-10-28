import math
import statistics
import numpy as np

CHANCE=1/64
FAILURE=CHANCE+.05
def select_learning_rate(candidates,L0):
    if not candidates or any(r['role']!='val' or r['length']!=L0 or r['position']!='sinusoidal' for r in candidates):
        raise ValueError('LR selection requires sinusoidal validation at L0 only')
    return max(candidates,key=lambda r:(r['accuracy'],-r['lr']))

def paired(a,b):
    if len(a)!=3 or len(b)!=3:raise ValueError('Exactly three paired seeds required')
    d=[x-y for x,y in zip(a,b)];mean=statistics.mean(d);sd=statistics.stdev(d)
    same=all(x>0 for x in d) or all(x<0 for x in d)
    return {'deltas':d,'mean':mean,'sample_std':sd,'same_nonzero_sign':same,
            'established':same and abs(mean)>=2*sd}

def slopes(lengths,values):
    if len(lengths)<2 or any(v<=0 for v in values):raise ValueError('Positive measurements required')
    x=np.log(lengths);y=np.log(values);slope,intercept=np.polyfit(x,y,1)
    return {'global':float(slope),'intercept':float(intercept),'longest_two':float((y[-1]-y[-2])/(x[-1]-x[-2])),
            'doubling_ratios':[{'L1':a,'L2':b,'ratio':v/u} for a,b,u,v in zip(lengths,lengths[1:],values,values[1:]) if b==2*a]}

def curve(rows,L0,absolute):
    baseline=next(r['accuracy'] for r in rows if r['length']==L0)
    result=[];boundary=None
    for r in sorted(rows,key=lambda r:r['length']):
        m=r.get('accuracy');oom=r.get('status')=='OOM'
        if r['length']>L0 and boundary is None and (oom or m is not None and m<=FAILURE):
            boundary={'length':r['length'],'reason':'OOM' if oom else 'accuracy_at_or_below_chance_plus_005'}
        result.append({**r,'ER':None if m is None else m/(baseline+1e-8),
                       'delta_pos':None if m is None else m-absolute[r['length']]})
    return {'points':result,'first_failure':boundary,'boundary_beyond_grid':boundary is None,
            'longest_measured_effective_length':max((r['length'] for r in rows if r.get('accuracy',0)>FAILURE),default=None)}
