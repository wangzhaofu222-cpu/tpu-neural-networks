import torch
from torch import nn
from src.multimodal.modality_mask import apply_mask,validate_mask,sample_modality_mask
from src.multimodal.alignment import masked_mean


def parameter_count(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matching_width(input_dim,classes,target):
    # One hidden layer: (input_dim+classes+1)*width + classes.
    step=input_dim+classes+1;q=(target-classes)/step
    candidates={max(1,int(q)),max(1,int(q)+1)};width=min(candidates,key=lambda w:(abs(step*w+classes-target),w))
    actual=step*width+classes
    return {'width':width,'target':int(target),'actual':actual,'relative_difference':abs(actual-target)/target,'integer_parameter_step':step,'nearest_integer_width':True}


class FusionClassifier(nn.Module):
    """Public modality API returns probabilities; packed harness input returns logits."""
    validation_macro_f1=True
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;self.dv=cfg['visual_dim'];self.da=cfg['audio_dim'];self.mode=cfg['strategy'];self.p_drop=cfg.get('p_drop',0.)
        width=cfg['hidden_width'];classes=cfg.get('classes',8);d=cfg.get('fusion_dim',128)
        if self.mode=='gated':
            self.video_projection=nn.Linear(self.dv,d,bias=False);self.audio_projection=nn.Linear(self.da,d,bias=False)
            self.gate=nn.Linear(self.dv+self.da,1 if cfg['gate']=='scalar' else d);input_dim=d
        elif self.mode=='early':input_dim=self.dv+self.da
        elif self.mode=='video':input_dim=self.dv
        elif self.mode=='audio':input_dim=self.da
        else:raise ValueError('Unregistered fusion strategy')
        self.head=nn.Sequential(nn.Linear(input_dim,width),nn.ReLU(),nn.Linear(width,classes))
        self.last_gate=None

    def logits(self,visual,audio,mask=(1,1),forced_gate=None):
        v,a=apply_mask(visual,audio,mask)
        if self.mode=='gated':
            g=torch.sigmoid(self.gate(torch.cat([v,a],-1))) if forced_gate is None else torch.full((len(v),1),forced_gate,device=v.device,dtype=v.dtype)
            z=g*torch.tanh(self.video_projection(v))+(1-g)*torch.tanh(self.audio_projection(a));self.last_gate=g.detach()
        else:z={'video':lambda:v,'audio':lambda:a,'early':lambda:torch.cat([v,a],-1)}[self.mode]();self.last_gate=None
        return self.head(z)

    def forward(self,x=None,*,visual=None,audio=None,mask=(1,1),align=None,forced_gate=None):
        if x is not None:
            v,a=x[:,:self.dv],x[:,self.dv:self.dv+self.da]
            m=sample_modality_mask(len(v),self.p_drop,v.device) if self.training and self.p_drop else mask
            if x.shape[1]==self.dv+self.da+2:
                available=x[:,-2:].bool();m=validate_mask(m,len(v),v.device)&available
                m=torch.where((m.sum(1)==0)[:,None],available,m)
            return self.logits(v,a,m)
        if visual is None or audio is None:raise ValueError('Provide both modality tensors and mask missing modalities inside model')
        presence=validate_mask(mask,len(visual),visual.device)
        def pool(values,valid,present):
            valid=torch.as_tensor(valid,device=values.device,dtype=torch.bool);n=valid.sum(1,keepdim=True)
            if ((n[:,0]==0)&present).any():raise ValueError('Empty present modality: explicitly mask absent frames')
            return (values*valid[:,:,None]).sum(1)/n.clamp_min(1)
        if visual.ndim==3:visual=pool(visual,align['visual_valid'],presence[:,0])
        if audio.ndim==3:audio=pool(audio,align['audio_valid'],presence[:,1])
        return self.logits(visual,audio,mask,forced_gate).softmax(-1)

    def parameter_breakdown(self):
        head=parameter_count(self.head);total=parameter_count(self)
        return {'frozen_encoders':0,'fusion_trainable':total-head,'head_trainable':head,'total_trainable':total,'encoders_external_frozen_cache':True}


def late_probabilities(p_video,p_audio,weight=.5,mask=(1,1)):
    if not 0<=weight<=1:raise ValueError('Invalid late fusion weight')
    m=validate_mask(mask,len(p_video),p_video.device);w=torch.full((len(p_video),1),weight,device=p_video.device,dtype=p_video.dtype)
    w=torch.where(~m[:,:1],torch.zeros_like(w),w);w=torch.where(~m[:,1:],torch.ones_like(w),w)
    return w*p_video+(1-w)*p_audio
