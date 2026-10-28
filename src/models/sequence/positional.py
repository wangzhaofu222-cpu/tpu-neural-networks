"""Absolute interpolation and Q/K rotary representations, Lab3 (3.5)--(3.9)."""
import math
import torch
from torch import nn

def sinusoid(positions,width):
    angles=positions.to(torch.float64)[:,None]/10000**(torch.arange(0,width,2,device=positions.device,dtype=torch.float64)/width)
    return torch.stack((angles.sin(),angles.cos()),-1).flatten(-2).float()

def rotate(x,positions):
    width=x.shape[-1]
    angles=positions.to(torch.float64)[:,None]/10000**(torch.arange(0,width,2,device=x.device,dtype=torch.float64)/width)
    c,s=angles.cos().to(x.dtype),angles.sin().to(x.dtype)
    a,b=x[...,0::2],x[...,1::2]
    return torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-2)

class Position(nn.Module):
    def __init__(self,mode,width,max_length,rule='P2',extension_seed=991):
        super().__init__();self.mode=mode;self.width=width;self.max_length=max_length;self.rule=rule;self.extension_seed=extension_seed
        if mode=='absolute':
            self.table=nn.Parameter(torch.empty(max_length,width));nn.init.normal_(self.table,std=.02)
        if mode not in {'none','absolute','sinusoidal','rope'}:raise ValueError(mode)
    def absolute(self,length,device):
        if length<=self.max_length:return self.table[:length]
        if self.rule=='P3':raise ValueError('P3: position beyond trained table')
        if self.rule=='P1':
            g=torch.Generator().manual_seed(self.extension_seed)
            extra=torch.randn(length-self.max_length,self.width,generator=g)*.02
            return torch.cat((self.table,extra.to(device)),0)
        if self.rule!='P2':raise ValueError(self.rule)
        t=torch.arange(length,device=device,dtype=torch.float64)*(self.max_length-1)/(length-1)
        a=(t-t.floor()).to(self.table.dtype)[:,None]
        return (1-a)*self.table[t.floor().long()]+a*self.table[t.ceil().long().clamp_max(self.max_length-1)]
    def forward(self,x):
        if self.mode=='absolute':return x+self.absolute(x.shape[1],x.device)
        if self.mode=='sinusoidal':return x+sinusoid(torch.arange(x.shape[1],device=x.device),self.width).to(x.dtype)
        return x
    def qk(self,q,k):
        if self.mode!='rope':return q,k
        p=torch.arange(q.shape[-2],device=q.device)
        return rotate(q,p),rotate(k,p)
