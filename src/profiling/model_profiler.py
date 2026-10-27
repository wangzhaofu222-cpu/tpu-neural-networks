import torch
from torch import nn


def measure_macs(model,x):
    counts={'conv':0,'linear':0,'attention':0};handles=[];batch=len(x)
    def hook(m,inputs,output):
        if isinstance(m,nn.Conv2d):
            counts['conv']+=output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]//batch
        elif isinstance(m,nn.Linear):
            counts['linear']+=output.numel()*m.in_features//batch
        elif isinstance(m,nn.MultiheadAttention):
            n=inputs[0].shape[1 if m.batch_first else 0];d=m.embed_dim
            counts['attention']+=4*n*d*d+2*n*n*d
    states={m:m.training for m in model.modules()}
    try:
        model.eval()
        for m in model.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear,nn.MultiheadAttention)):handles.append(m.register_forward_hook(hook))
        with torch.no_grad():model(x)
    finally:
        for h in handles:h.remove()
        for m,state in states.items():m.training=state
    return {'MACs':sum(counts.values()),'components':counts,'convention':'one multiply-accumulate; excludes normalization, activation, softmax, bias additions'}


def analytic_macs(cfg):
    if cfg['name']=='cifar_resnet18':
        total=32*32*64*3*9;size=32;incoming=64
        for stage,width in enumerate([64,128,256,512]):
            for block in range(2):
                stride=2 if stage>0 and block==0 else 1
                size//=stride
                total+=size*size*(width*incoming*9+width*width*9)
                if stride!=1 or incoming!=width:total+=size*size*incoming*width
                incoming=width
        return {'MACs':total+512*cfg['classes']}
    d,p,L=cfg['hidden_dim'],cfg['patch_size'],cfg['num_layers']
    n=(cfg['image_size']//p)**2+1
    block=12*n*d*d+2*n*n*d
    return {'MACs':(n-1)*d*3*p*p+L*block+d*cfg['classes'],'tokens':n,'block_MACs':block,'attention_products_MACs':2*n*n*d}


def resource_profile(model,cfg,device):
    from src.profile import profile_model
    measured=measure_macs(model,torch.zeros(1,3,32,32,device=device))
    expected=analytic_macs(cfg)
    timed=profile_model(model,[1,128],device,warmup=10,repeats=30)
    return {**timed,**measured,'analytic':expected,'MAC_relative_error':abs(measured['MACs']/expected['MACs']-1),
         'latency_B1_seconds':timed['measurements'][0]['median_seconds'],
         'throughput_B128_per_second':128/timed['measurements'][1]['median_seconds'],
         'precision':'float32 eval','comparison_mode':'param_matched'}
