import torch
import torch.nn.functional as F
from src.metrics.objective import correct_count,objective


@torch.no_grad()
def evaluate_loader(model,loader,device):
    was_training = model.training
    model.eval()
    loss_sum,correct,count = 0.,0,0
    macro_mode=getattr(model,'validation_macro_f1',False);macro_logits=[];macro_labels=[]
    try:
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            logits = model(x)
            if macro_mode:macro_logits.append(logits.detach().cpu());macro_labels.append(y.detach().cpu())
            loss_sum += objective(logits,y,reduction='sum').item()
            hits,targets=correct_count(logits,y)
            correct += hits
            count += targets
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError('Empty evaluation role')
    result={'loss':loss_sum/count,'accuracy':correct/count,'count':count}
    if macro_mode:
        from src.diagnostics.contribution import classification
        result['macro_f1']=classification(torch.cat(macro_logits).softmax(1).numpy(),torch.cat(macro_labels).numpy())['macro_f1']
    return result
