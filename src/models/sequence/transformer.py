import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from src.models.sequence.positional import Position

class Attention(nn.Module):
    def __init__(self,width,heads,backend):
        super().__init__();self.heads=heads;self.backend=backend
        self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
    def forward(self,x,position,mask=None,return_attention=False):
        b,l,d=x.shape;q,k,v=self.qkv(x).reshape(b,l,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        q,k=position.qk(q,k)
        if self.backend=='explicit' or return_attention:
            scores=(q@k.transpose(-1,-2))/math.sqrt(d//self.heads)
            if mask is not None:scores=scores.masked_fill(~mask,float('-inf'))
            a=scores.softmax(-1);z=a@v
        else:
            if self.backend!='flash':raise ValueError(self.backend)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,is_causal=False)
            a=None
        return self.out(z.transpose(1,2).reshape(b,l,d)),a

class Block(nn.Module):
    def __init__(self,width,heads,ffn,backend):
        super().__init__();self.ln1=nn.LayerNorm(width);self.ln2=nn.LayerNorm(width)
        self.attention=Attention(width,heads,backend);self.mlp=nn.Sequential(nn.Linear(width,ffn),nn.GELU(),nn.Linear(ffn,width))
    def forward(self,x,position,mask=None,return_attention=False):
        y,a=self.attention(self.ln1(x),position,mask,return_attention);x=x+y
        return x+self.mlp(self.ln2(x)),a

class SequenceTransformer(nn.Module):
    def __init__(self,cfg):
        super().__init__();d=cfg['width'];self.config=cfg
        # Identical backbone initialization across PE modes: table is initialized last.
        self.embedding=nn.Embedding(69,d)
        self.blocks=nn.ModuleList([Block(d,cfg['heads'],cfg['ffn'],cfg['backend']) for _ in range(cfg['depth'])])
        self.norm=nn.LayerNorm(d);self.output=nn.Linear(d,64)
        self.position=Position(cfg['position'],d,cfg['max_length'],cfg.get('absolute_rule','P2'))
    def forward(self,x,mask=None,return_attention=False):
        z=self.position(self.embedding(x));matrices=[]
        for block in self.blocks:
            z,a=block(z,self.position,mask,return_attention)
            if return_attention:matrices.append(a)
        logits=self.output(self.norm(z)).transpose(1,2)
        return (logits,matrices) if return_attention else logits
