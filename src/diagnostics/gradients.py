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
        finite = all(bool(torch.isfinite(g).all()) for g in grads) and all(bool(torch.isfinite(p).all()) for p in params)
        if not finite:
            rows[name] = {'numerical_failure': True, 'g': None}
            continue
        norm = math.sqrt(sum(g.square().sum().item() for g in grads))
        pn = math.sqrt(sum(p.detach().double().square().sum().item() for p in params))
        if not math.isfinite(norm) or not math.isfinite(pn):
            rows[name] = {'numerical_failure': True, 'g': None}
            continue
        rows[name] = {'g': norm/math.sqrt(sum(p.numel() for p in params)), 'gradient_norm': norm,
                      'relative_gradient': norm/(pn+1e-12), 'unused': False, 'numerical_failure': False}
        rows[name]['per_parameter'] = {}
        for key, parameter in layer.named_parameters(recurse=False):
            if parameter.grad is not None:
                gn = parameter.grad.detach().double().norm().item()
                rows[name]['per_parameter'][key] = {'gradient_norm': gn, 'g': gn/math.sqrt(parameter.numel())}
    return rows
