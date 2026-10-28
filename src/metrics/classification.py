import torch
import torch.nn.functional as F
from src.metrics.objective import correct_count


@torch.no_grad()
def evaluate_loader(model,loader,device):
    was_training = model.training
    model.eval()
    loss_sum,correct,count = 0.,0,0
    try:
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            logits = model(x)
            loss_sum += F.cross_entropy(logits,y,reduction='sum').item()
            hits,targets=correct_count(logits,y)
            correct += hits
            count += targets
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError('Empty evaluation role')
    return {'loss':loss_sum/count,'accuracy':correct/count,'count':count}
