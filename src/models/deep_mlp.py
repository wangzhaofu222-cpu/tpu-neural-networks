import math
import torch
from torch import nn


def normalization(name, width):
    return {'none': nn.Identity, 'batchnorm': lambda: nn.BatchNorm1d(width),
            'layernorm': lambda: nn.LayerNorm(width)}[name]()


def initialize(model, scheme, sigma):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            fan_out, fan_in = layer.weight.shape
            std = {'he': math.sqrt(2/fan_in), 'xavier': math.sqrt(2/(fan_in+fan_out)),
                   'gaussian': sigma}[scheme]
            nn.init.normal_(layer.weight, std=std)
            nn.init.zeros_(layer.bias)


class HiddenLayer(nn.Module):
    def __init__(self, inputs, width, activation, norm):
        super().__init__()
        self.linear = nn.Linear(inputs, width)
        self.norm = normalization(norm, width)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()

    def forward(self, x):
        return self.activation(self.norm(self.linear(x)))


class DeepMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.get('normalization_position', 'before_activation') != 'before_activation':
            raise ValueError('Only normalization before activation is supported')
        self.layers = nn.ModuleList([HiddenLayer(cfg.get('input_features', 3072) if i == 0 else cfg['width'],
                      cfg['width'], cfg['activation'], cfg['normalization']) for i in range(cfg['depth'])])
        self.head = nn.Linear(cfg['width'], cfg['classes'])
        initialize(self, cfg['initialization'], cfg['sigma'])

    def diagnostic_layers(self):
        return {f'h{i+1}': layer for i, layer in enumerate(self.layers)}

    def forward_features(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        return self.head(self.forward_features(x))
