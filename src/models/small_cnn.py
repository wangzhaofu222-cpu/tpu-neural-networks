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
    if cfg['name']=='graph_network':
        from src.models.graph import GraphNetwork
        return GraphNetwork(cfg)
    if cfg['name']=='simclr':
        from src.ssl.simclr import SimCLR
        return SimCLR(cfg)
    if cfg['name']=='linear_probe':
        return nn.Linear(cfg['input_dim'],cfg['classes'])
    if cfg['name']=='sequence_transformer':
        from src.models.sequence.transformer import SequenceTransformer
        return SequenceTransformer(cfg)
    if cfg['name']=='cifar_resnet18':
        from src.models.vision.resnet import CifarResNet18
        return CifarResNet18(cfg['classes'])
    if cfg['name']=='cifar_vit':
        from src.models.vision.vit import CifarViT
        return CifarViT(cfg)
    if cfg['name'] == 'deep_mlp':
        from src.models.deep_mlp import DeepMLP
        return DeepMLP(cfg)
    if cfg['name'] == 'residual_mlp':
        from src.models.residual_mlp import ResidualMLP
        return ResidualMLP(cfg)
    if cfg['name'] != 'small_cnn':
        raise ValueError('Unregistered model: '+cfg['name'])
    return SmallCNN(cfg['width'],cfg['depth'],cfg['classes'])
