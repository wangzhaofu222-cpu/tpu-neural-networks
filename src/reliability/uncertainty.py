import torch


def uncertainty(logits, temperature=1.):
    if temperature<=0:raise ValueError('Positive temperature required')
    z=torch.as_tensor(logits,dtype=torch.float64)/temperature
    lp=z.log_softmax(1);p=lp.exp();top=p.topk(2,dim=1).values
    return {'msp':1-top[:,0],'entropy':-torch.special.xlogy(p,p).sum(1),'margin':1-top[:,0]+top[:,1]}
