import os
from pathlib import Path
import torch
from src.utils.reproducibility import rng_state,restore_rng


def save_checkpoint(path,model,optimizer,scheduler,scaler,epoch,global_step,generator,**metadata):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
             'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
             'epoch':epoch,'global_step':global_step, **rng_state(generator), **metadata}
    temporary = path.with_suffix('.tmp')
    torch.save(state,temporary)
    os.replace(temporary,path)


def load_checkpoint(path,model,optimizer=None,scheduler=None,scaler=None,generator=None):
    # Only load trusted locally generated training checkpoints (includes NumPy/Python RNG).
    state = torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state,generator)
    return state
