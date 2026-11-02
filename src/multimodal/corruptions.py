import math
import torch


def add_noise(waveform,snr_db,seed=1):
    x=torch.as_tensor(waveform,dtype=torch.float64);power=x.square().sum()
    if not len(x) or power<=0:raise ValueError('SNR undefined for empty or silent audio; no noise result fabricated')
    g=torch.Generator().manual_seed(seed);noise=torch.randn(x.shape,generator=g,dtype=x.dtype)
    gamma=torch.sqrt(power/(10**(snr_db/10)*noise.square().sum()));delta=gamma*noise;y=x+delta
    measured=10*torch.log10(power/(y-x).square().sum()).item()
    if abs(measured-snr_db)>.1:raise RuntimeError('SNR tolerance exceeded')
    return y,{'target_snr_db':snr_db,'measured_snr_db':measured,'gamma':float(gamma),'signal_power':float(power/len(x)),'noise_power':float(delta.square().mean()),'seed':seed}


def time_mask(mel,level,seed=1):
    if level not in range(6):raise ValueError('Invalid time mask level')
    if level==0:return mel
    y=mel.clone();width=max(1,round(y.shape[-1]*.05*level));g=torch.Generator().manual_seed(seed);start=int(torch.randint(y.shape[-1]-width+1,(),generator=g));y[...,start:start+width]=0
    return y


def frame_dropout(valid,level,seed=1):
    if level not in range(6):raise ValueError('Invalid frame dropout level')
    if level==0:return valid
    g=torch.Generator().manual_seed(seed);m=valid.clone() & (torch.rand(valid.shape,generator=g)>=.1*level)
    return m
