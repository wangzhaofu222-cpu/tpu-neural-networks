import math
import torch
from torch import nn


def gradient_statistics(model):
    rows = {}
    for name, layer in model.named_modules():
        params = list(layer.parameters(recurse=False))
        if not params:
            continue
        grads = [p.grad.detach().double() for p in params if p.grad is not None]
        if not grads:
            rows[name] = {'g': None, 'gradient_norm': None, 'relative_gradient': None, 'unused': True,
                          'numerical_failure': False}
            continue
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        if not finite:
            rows[name] = {'numerical_failure': True, 'g': None}
            continue
        norm = math.sqrt(sum(g.square().sum().item() for g in grads))
        pn = math.sqrt(sum(p.detach().double().square().sum().item() for p in params))
        rows[name] = {'g': norm/math.sqrt(sum(p.numel() for p in params)), 'gradient_norm': norm,
                      'relative_gradient': norm/(pn+1e-12), 'unused': False, 'numerical_failure': False}
    return rows
