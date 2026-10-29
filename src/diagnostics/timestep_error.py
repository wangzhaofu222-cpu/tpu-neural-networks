import torch
from src.diffusion.forward import q_sample
from src.diffusion.parameterizations import EpsilonParameterization


@torch.no_grad()
def timestep_errors(eps_fn, exact, schedule, val, noise, times):
    rows=[]
    param=EpsilonParameterization(schedule)
    for t in times:
        xt,_=q_sample(val,t,schedule,noise)
        prediction=eps_fn(xt,torch.tensor(t,device=val.device))
        star=exact(xt,t)
        residual=(noise-prediction).double()
        irreducible=(noise-star).double()
        network=(star-prediction).double()
        cross=(2*irreducible*network).mean(1)
        rows.append({'t':t,'E_epsilon':float(residual.square().mean()),
            'E_x0':float((val-param.to_x0(prediction,xt,t)).double().square().mean()),
            'E_epsilon_star':float(irreducible.square().mean()),'E_net':float(network.square().mean()),
            'cross_term':float(cross.mean()),'cross_term_SEM':float(cross.std()/len(cross)**.5),
            'SNR':float(schedule.snr[t]),'network_x0_amplification':float(network.square().mean()/schedule.snr[t])})
    return rows
