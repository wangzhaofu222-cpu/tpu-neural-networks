import torch
from torchvision.models.vision_transformer import VisionTransformer


def patchify(x,p):
    b,c,h,w=x.shape
    if h%p or w%p: raise ValueError('Image dimensions must be divisible by patch size')
    return x.reshape(b,c,h//p,p,w//p,p).permute(0,2,4,1,3,5).reshape(b,(h//p)*(w//p),c*p*p)


def unpatchify(x,p,h,w):
    b,n,d=x.shape;c=d//(p*p)
    if n!=(h//p)*(w//p): raise ValueError('Wrong number of patches')
    return x.reshape(b,h//p,w//p,c,p,p).permute(0,3,1,4,2,5).reshape(b,c,h,w)


class CifarViT(VisionTransformer):
    def __init__(self,cfg):
        super().__init__(image_size=cfg['image_size'],patch_size=cfg['patch_size'],
              num_layers=cfg['num_layers'],num_heads=cfg['num_heads'],hidden_dim=cfg['hidden_dim'],
              mlp_dim=cfg['mlp_dim'],num_classes=cfg['classes'])

    def forward_features(self,x,return_intermediate=False):
        x=self._process_input(x)
        x=torch.cat([self.class_token.expand(len(x),-1,-1),x],dim=1)
        if not return_intermediate: return self.encoder(x)[:,0]
        x=self.encoder.dropout(x+self.encoder.pos_embedding);layers={}
        for i,layer in enumerate(self.encoder.layers,1):
            x=layer(x);layers[f'block{i}']=self.encoder.ln(x)[:,0]
        return layers

    def forward(self,x):
        return self.heads(self.forward_features(x))
