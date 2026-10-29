import math
import torch

def reflect(x,boundary):
    if boundary is None:return x
    low=x.new_tensor([b[0] for b in boundary]);high=x.new_tensor([b[1] for b in boundary]);width=high-low
    if (width<=0).any():raise ValueError('Invalid domain')
    return low+width-torch.abs(torch.remainder(x-low,2*width)-width)

def langevin_sample(energy_fn,x0,eta,temperature,steps,boundary,generator=None,return_diagnostics=False):
    if eta<=0 or temperature<0 or steps<0:raise ValueError('Invalid Langevin parameters')
    x=x0.detach().clone();trajectory=[x.clone()];energies=[];outside=0;proposals=0
    sigma=math.sqrt(2*eta*temperature)
    for _ in range(steps):
        with torch.enable_grad():
            x=x.detach().requires_grad_(True);e=energy_fn(x);grad=torch.autograd.grad(e.sum(),x,create_graph=False)[0]
        energies.append(float(e.detach().mean()))
        noise=torch.randn(x.shape,dtype=x.dtype,generator=generator).to(x.device)
        proposal=x.detach()-eta*grad.detach()+sigma*noise
        if boundary is not None:
            low=x.new_tensor([b[0] for b in boundary]);high=x.new_tensor([b[1] for b in boundary])
            outside+=int(((proposal<low)|(proposal>high)).any(-1).sum());proposals+=len(x)
        x=reflect(proposal,boundary).detach();trajectory.append(x.clone())
    with torch.no_grad():energies.append(float(energy_fn(x).mean()))
    traj=torch.stack(trajectory)
    if return_diagnostics:
        return x,traj,{'eta':eta,'temperature':temperature,'sigma':sigma,'sigma_squared':2*eta*temperature,
            'DK':float((x-x0).norm(dim=-1).mean()),'energy_curve':energies,
            'pre_reflection_outside_proposal_fraction':outside/proposals if proposals else 0.,
            'finite':bool(torch.isfinite(traj).all())}
    return x,traj

class ReplayBuffer:
    def __init__(self,capacity=10000,seed=1,restart=.05):
        self.generator=torch.Generator().manual_seed(seed);self.restart=float(restart)
        self.points=torch.randn(capacity,2,generator=self.generator)
    def sample(self,n):
        idx=torch.randint(len(self.points),(n,),generator=self.generator)
        out=self.points[idx].clone();mask=torch.rand(n,generator=self.generator)<self.restart
        out[mask]=torch.randn(int(mask.sum()),2,generator=self.generator);return out
    def update(self,points):
        # A permutation selects distinct random cells per update, avoiding ambiguous duplicate writes.
        n=len(points)
        if n>len(self.points):raise ValueError('Batch exceeds replay capacity')
        idx=torch.randperm(len(self.points),generator=self.generator)[:n]
        self.points[idx]=points.detach().cpu().to(self.points)
    def state_dict(self):return {'points':self.points.clone(),'generator_state':self.generator.get_state().clone(),'restart':self.restart}
    def load_state_dict(self,state):
        self.points=state['points'].clone();self.generator.set_state(state['generator_state'].cpu());self.restart=state['restart']
