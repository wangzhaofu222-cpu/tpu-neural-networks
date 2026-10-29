import torch
from src.diffusion.schedules import make_schedule
from src.diffusion.forward import q_sample
from src.diffusion.objectives import simple_loss


class DiffusionState:
    def __init__(self, cfg):
        self.schedule = make_schedule(cfg['diffusion']).to(cfg['runtime']['device'])
        self.generator = torch.Generator(device=cfg['runtime']['device']).manual_seed(cfg['experiment']['seed']+71000)
        self.val_seed = 72000
        self.last_trajectory = None

    def state_dict(self):
        return {'training_noise_rng': self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state['training_noise_rng'].cpu())

    def objective(self, model, x):
        t = torch.randint(1, self.schedule.T+1, (len(x),), device=x.device, generator=self.generator)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=self.generator)
        xt, _ = q_sample(x, t, self.schedule, noise)
        loss = simple_loss(model(xt, t), noise)
        return loss, {'Lsimple': float(loss.detach()), 't_mean': float(t.float().mean()),
                      'noise_checksum_sum': float(noise.sum()), 'loss_definition': 'mean_batch_sum_coordinates'}

    @torch.no_grad()
    def evaluate(self, model, loader, device):
        was = model.training
        model.eval()
        g = torch.Generator(device=device).manual_seed(self.val_seed)
        total, count = 0., 0
        for x, _ in loader:
            x = x.to(device)
            t = torch.randint(1, self.schedule.T+1, (len(x),), device=device, generator=g)
            noise = torch.randn(x.shape, device=device, generator=g)
            xt, _ = q_sample(x, t, self.schedule, noise)
            total += (model(xt, t)-noise).double().square().sum().item()
            count += len(x)
        model.train(was)
        return {'loss': total/count, 'accuracy': None, 'metric_name': 'Lsimple_not_NLL'}
