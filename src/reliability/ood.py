import numpy as np


def detection_metrics(U, ood_flags):
    """No semantic class-label argument; bool flags only, OOD is positive."""
    u=np.asarray(U,dtype=float);y=np.asarray(ood_flags)
    if y.dtype!=np.bool_ or u.shape!=y.shape or len(np.unique(y))!=2 or not np.isfinite(u).all():
        raise ValueError('Only uncertainty and binary boolean OOD flags accepted')
    order=np.argsort(-u,kind='stable');s=u[order];flags=y[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.cumsum(flags)[ends];fp=(ends+1)-tp
    tpr=np.r_[0.,tp/y.sum()];fpr=np.r_[0.,fp/(~y).sum()];threshold=np.r_[np.inf,s[ends]]
    idx=np.flatnonzero(tpr>=.95)[0]
    auroc=float(np.trapezoid(tpr,fpr));aupr=float(np.sum(np.diff(tpr)*tp/(tp+fp)))
    return {'AUROC':auroc,'AUPR':aupr,
            'FPR95':float(fpr[idx]),'threshold95':float(threshold[idx]),'positive_class':'OOD',
            'N_ID':int((~y).sum()),'N_OOD':int(y.sum()),'OOD_fraction':float(y.mean()),'threshold_rule':'U >= threshold, first ROC TPR >= 0.95'}
