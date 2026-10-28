import numpy as np
from src.data.splits import train_subset
from src.diagnostics.extrapolation import paired

def label_subsets(labels,indices,fractions,seed):
    ids=np.asarray(indices);local_labels=np.asarray(labels)[ids]
    # Even class discovery is restricted totrain; never inspectcal/val labels here.
    result={str(r):sorted(ids[train_subset(local_labels,range(len(ids)),r,seed)].tolist()) for r in fractions}
    ordered=sorted(fractions)
    assert all(set(result[str(a)])<=set(result[str(b)]) for a,b in zip(ordered,ordered[1:]))
    return result

def paired_gain(ssl,random):return paired(ssl,random)

def classify_ablation(gain,delta,collapse):
    if collapse:return 'critical_collapse_observed'
    if gain>0 and delta>=gain/2:return 'possibly_critical_single_seed'
    if gain>0 and delta<=-gain/2:return 'possibly_harmful_single_seed'
    return 'indistinguishable_from_noise_single_seed'
