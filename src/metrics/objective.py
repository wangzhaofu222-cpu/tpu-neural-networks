"""Common classification/token objective; sequence non-targets use ignore_index=-100."""
import torch.nn.functional as F

def objective(logits,y,reduction='mean'):
    if logits.ndim==3 and y.ndim==2:
        # Flatten token classification to deterministic CUDA NLL, avoiding NLL2d.
        return F.cross_entropy(logits.transpose(1,2).reshape(-1,logits.shape[1]),y.reshape(-1),reduction=reduction)
    return F.cross_entropy(logits,y,reduction=reduction)

def correct_count(logits,y):
    if y.ndim==2 and logits.ndim==3:
        mask=y!=-100
        return ((logits.argmax(1)==y)&mask).sum().item(),mask.sum().item()
    correct=(logits.argmax(1)==y).sum().item() if y.ndim==1 else y.gather(1,logits.argmax(1)[:,None]).sum().item()
    return correct,len(y)
