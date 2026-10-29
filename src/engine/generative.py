"""Generative objective/checkpoint adapters for the existing ExperimentRunner loop."""
import math
import torch
from src.generative.samplers import ReplayBuffer,langevin_sample
from src.generative.distributions import distribution

class GenerativeState:
    def __init__(self,cfg):
        self.cfg=cfg;self.target=distribution(cfg['data']['target'])
        self.replay=ReplayBuffer(10000,cfg['experiment']['seed'],.05) if cfg['model']['name']=='energy_model' else None
        self.last_trajectory=None
    def state_dict(self):return {'replay':self.replay.state_dict() if self.replay else None}
    def load_state_dict(self,state):
        if self.replay:self.replay.load_state_dict(state['replay'])
    def objective(self,model,x):
        if self.replay is None:
            nll=-model.log_prob(x);return nll.mean(),{'exact_train_NLL':float(nll.detach().mean())}
        c=self.cfg['sampling'];start=self.replay.sample(len(x)).to(x.device)
        negative,traj,diag=langevin_sample(model,start,c['eta'],c['temperature'],c['steps'],self.target.domain,self.replay.generator,True)
        self.replay.update(negative);self.last_trajectory=traj.detach().cpu()
        positive_energy=model(x);negative_energy=model(negative.detach())
        loss=positive_energy.mean()-negative_energy.mean()
        coefficient=self.cfg['generative']['energy_regularization']
        if coefficient:loss=loss+coefficient*(positive_energy.square().mean()+negative_energy.square().mean())
        from src.diagnostics.mcmc import mode_switches
        diag.update(positive_energy=float(positive_energy.detach().mean()),negative_energy=float(negative_energy.detach().mean()),
                    energy_regularization=coefficient,**mode_switches(traj,self.target))
        return loss,diag

@torch.no_grad()
def evaluate(model,loader,device):
    from src.generative.flows import NormalizingFlow
    was=model.training;model.eval();total=0.;n=0
    try:
        for x,_ in loader:
            x=x.to(device);v=-model.log_prob(x) if isinstance(model,NormalizingFlow) else model(x)
            total+=v.double().sum().item();n+=len(x)
    finally:model.train(was)
    return {'loss':total/n,'accuracy':None,'metric_name':'exact_NLL' if isinstance(model,NormalizingFlow) else 'mean_energy_not_NLL'}
