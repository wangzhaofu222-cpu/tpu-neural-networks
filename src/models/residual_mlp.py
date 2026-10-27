import math
from torch import nn
from src.models.deep_mlp import HiddenLayer, initialize


class ResidualBlock(nn.Module):
    def __init__(self, width, activation='relu', norm='none', alpha=1.):
        super().__init__()
        self.branch = nn.Sequential(HiddenLayer(width, width, activation, norm), nn.Linear(width, width))
        self.alpha = alpha

    def forward(self, x):
        return x + self.alpha*self.branch(x)


class ResidualMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.get('normalization_position', 'before_activation') != 'before_activation':
            raise ValueError('Only normalization before activation is supported')
        self.input = HiddenLayer(cfg.get('input_features', 3072), cfg['width'], cfg['activation'], cfg['normalization'])
        alpha = 1/math.sqrt(cfg['depth']) if cfg['alpha'] == 'inverse_sqrt_depth' else float(cfg['alpha'])
        self.blocks = nn.ModuleList([ResidualBlock(cfg['width'], cfg['activation'], cfg['normalization'], alpha)
                                    for _ in range(cfg['depth'])])
        self.head = nn.Linear(cfg['width'], cfg['classes'])
        initialize(self, cfg['initialization'], cfg['sigma'])

    def diagnostic_layers(self):
        return {'h0': self.input, **{f'h{i+1}': b for i, b in enumerate(self.blocks)}}

    def forward_features(self, x):
        x = self.input(x.flatten(1))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x):
        return self.head(self.forward_features(x))
