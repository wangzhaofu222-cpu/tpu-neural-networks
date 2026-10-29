import torch
from src.data.gmm2d import standardize,inverse_standardize


def trajectory_modes(saved,schedule,distribution,stats):
    final=distribution.mode_id(inverse_standardize(saved[0].double(),stats))
    centers=standardize(distribution.centers,stats)
    rows=[]
    for fraction in [.8,.6,.4,.2,.05]:
        t=round(fraction*schedule.T)
        labels=torch.cdist(saved[t].double(),schedule.alpha_bar[t].sqrt().cpu()*centers).argmin(1)
        valid=final<distribution.M
        rows.append({'t':t,'fraction':fraction,'agreement_all':float((labels==final).double().mean()),
            'agreement_inlier_final':float((labels[valid]==final[valid]).double().mean()) if valid.any() else None,
            'final_outlier_fraction':float((~valid).double().mean()),
            'per_final_mode':[float((labels[final==m]==m).double().mean()) if (final==m).any() else None for m in range(distribution.M)]})
    return rows
