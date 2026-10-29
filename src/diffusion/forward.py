import torch
from .schedules import extract


def q_sample(x0, t, schedule, noise=None):
    noise = torch.randn_like(x0) if noise is None else noise
    if noise.shape != x0.shape:
        raise ValueError('Noise shape mismatch')
    a = extract(schedule.alpha_bar, t, x0)
    return a.sqrt()*x0+(1-a).sqrt()*noise, noise
