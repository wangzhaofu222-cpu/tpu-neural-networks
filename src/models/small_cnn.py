import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self,width,depth,classes=10):
        super().__init__()
        layers = []
        channels = 3
        for i in range(depth):
            layers.extend([nn.Conv2d(channels,width,3,padding=1),nn.ReLU()])
            if i < 3:
                layers.append(nn.MaxPool2d(2))
            channels = width
        self.features = nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head = nn.Linear(width,classes)

    def forward_features(self,x):
        return self.features(x)

    def forward(self,x):
        return self.head(self.forward_features(x))


def build_model(cfg):
    if cfg['name'] == 'deep_mlp':
        from src.models.deep_mlp import DeepMLP
        return DeepMLP(cfg)
    if cfg['name'] == 'residual_mlp':
        from src.models.residual_mlp import ResidualMLP
        return ResidualMLP(cfg)
    if cfg['name'] != 'small_cnn':
        raise ValueError('Unregistered model: '+cfg['name'])
    return SmallCNN(cfg['width'],cfg['depth'],cfg['classes'])
