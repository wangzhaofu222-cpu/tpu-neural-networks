import torch


def statistics(tensor, activation=None):
    x = tensor.detach().to(torch.float64)
    finite = bool(torch.isfinite(x).all())
    if not finite:
        return {'numerical_failure': True}
    variance = x.var(unbiased=False).item()
    return {'mean': x.mean().item(), 'std': variance**.5, 'variance': variance,
            'q': x.square().mean().item(), 'min': x.min().item(), 'max': x.max().item(),
            'zero_fraction': (x == 0).double().mean().item() if activation == 'relu' else None,
            'saturated_fraction': ((1-x.square()).abs() < .1).double().mean().item() if activation == 'tanh' else None,
            'numerical_failure': False}


class ActivationMonitor:
    """Observational hooks: never return replacement outputs or gradients."""
    def __init__(self, layers, activation='relu'):
        self.rows, self.handles = {}, []
        for name, layer in layers.items():
            def hook(module, inputs, output, name=name):
                kind = None if hasattr(module, 'branch') else activation
                self.rows[name] = statistics(output, kind)
                if output.requires_grad:
                    def backward(gradient):
                        self.rows[name]['delta'] = statistics(gradient)
                    output.register_hook(backward)
            self.handles.append(layer.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
