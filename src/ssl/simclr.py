import torch
from torch import nn
from src.models.vision.resnet import CifarResNet18

class SimCLR(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.encoder=CifarResNet18();self.encoder.fc=nn.Identity()
        self.projector=nn.Sequential(nn.Linear(512,cfg['projection_hidden'],bias=False),nn.ReLU(),nn.Linear(cfg['projection_hidden'],cfg['projection_dim'],bias=False))
        self.classifier=nn.Linear(512,10);self.mode=cfg['mode']
    def forward_features(self,x):return self.encoder.forward_features(x)
    def representations(self,x):
        h=self.forward_features(x);return h,torch.nn.functional.normalize(self.projector(h),dim=1)
    def forward(self,x):
        if x.ndim==5:
            a,b=x[:,0],x[:,1]
            # One simultaneous2N pass preserves global BatchNorm, unlike microbatch approximations.
            z=self.projector(self.forward_features(torch.cat((a,b),0)))
            return z[:len(a)],z[len(a):]
        return self.classifier(self.forward_features(x))
