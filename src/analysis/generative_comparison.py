import math
import torch
from src.generative.ebm import EnergyModel
from src.generative.flows import NormalizingFlow

def match_parameters(flow_depth,cap):
    flow_count=sum(p.numel() for p in NormalizingFlow({'depth':flow_depth,'width':64,'cap':cap}).parameters())
    base_count=sum(p.numel() for p in EnergyModel({'width':128}).parameters())
    if abs(base_count-flow_count)/min(base_count,flow_count)<=.2:width=128
    else:width=min(range(8,513),key=lambda w:abs((2*w*w+6*w+1)-flow_count))
    ebm_count=sum(p.numel() for p in EnergyModel({'width':width}).parameters())
    return {'ebm_width':width,'flow_width':64,'flow_depth':flow_depth,'ebm_parameters':ebm_count,'flow_parameters':flow_count,
        'relative_gap_over_smaller':abs(ebm_count-flow_count)/min(ebm_count,flow_count),'original_EBM_width':128,
        'rule':'retain width128 if within20%, otherwise integer width minimizing gap; only EBM width changes'}

def nearest_centers(target):
    c=target.centers;d=torch.cdist(c,c);d.fill_diagonal_(float('inf'))
    pairs=[(float(d[i,j]),i,j) for i in range(len(c)) for j in range(i+1,len(c))]
    # Quantized distance ties broken by original center index, not model outcomes.
    _,i,j=min(pairs,key=lambda t:(round(t[0],10),t[1],t[2]))
    return c[i],c[j],(c[i]+c[j])/2,[i,j]
