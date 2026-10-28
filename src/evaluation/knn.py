import torch
from torch.nn.functional import normalize,one_hot

@torch.no_grad()
def weighted_knn(train_h,train_y,query_h,k=200,tau=.07,reference_role='train',chunk=256):
    if reference_role!='train':raise PermissionError('kNN reference must contain only train images')
    if k>len(train_h) or tau<=0:raise ValueError('Invalid kNN parameters')
    train=normalize(train_h.float(),dim=1);pred=[]
    for q in query_h.split(chunk):
        values,indices=(normalize(q.float(),dim=1)@train.T).topk(k,dim=1)
        # Subtracting the per-query maximum preserves the weighted-vote argmax.
        weights=((values-values.max(1,keepdim=True).values)/tau).exp()
        votes=(one_hot(train_y[indices],10)*weights[:,:,None]).sum(1);pred.append(votes.argmax(1))
    return torch.cat(pred)
