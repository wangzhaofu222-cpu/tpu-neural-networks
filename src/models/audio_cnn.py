import torch
from torch import nn


class AudioCNN(nn.Module):
    validation_macro_f1=True
    def __init__(self,cfg):
        super().__init__();layers=[];ch=1
        for width in [16,32,64,128]:layers.append(nn.Sequential(nn.Conv2d(ch,width,3,padding=1),nn.ReLU(),nn.MaxPool2d((2,1))));ch=width
        self.blocks=nn.ModuleList(layers);self.head=nn.Linear(128,cfg.get('classes',8));self.r=1
    def forward_features(self,x):
        valid=x[:,1,0,:].bool();h=x[:,:1]
        for block in self.blocks:h=block(h)*valid[:,None,None,:]
        return h.mean(2).transpose(1,2),valid
    def forward(self,x):
        z,valid=self.forward_features(x);pooled=(z*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        return self.head(pooled)
