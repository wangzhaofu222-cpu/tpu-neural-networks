"""Unequal GMM, train-only standardization, and differentiable exact noisy density."""
import hashlib
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.utils.io import digest


class RareGMM:
    def __init__(self, M, rho, pi0):
        self.M, self.rho, self.pi0 = M, rho, pi0
        angles = torch.arange(M, dtype=torch.float64)*2*math.pi/M
        self.centers = torch.stack([angles.cos(), angles.sin()], 1)
        self.weights = torch.full((M,), (1-pi0)/(M-1), dtype=torch.float64)
        self.weights[0] = pi0

    def sample(self, n, seed, return_components=False):
        g = torch.Generator().manual_seed(seed)
        labels = torch.multinomial(self.weights, n, replacement=True, generator=g)
        x = self.centers[labels]+self.rho*torch.randn(n, 2, generator=g, dtype=torch.float64)
        return (x, labels) if return_components else x

    def mode_id(self, x):
        distance = torch.cdist(x.double(), self.centers.to(x.device))
        d, m = distance.min(1)
        return torch.where(d < 3*self.rho, m, self.M)


def target(cfg):
    d = cfg['data']
    return RareGMM(d['M'], d['rho'], d['pi0'])


def roles_and_stats(cfg):
    dist = target(cfg)
    seeds = cfg['data']['seeds']
    roles = {k: dist.sample(n, seeds[k]) for k, n in [('train', 20000), ('val', 5000), ('test', 5000)]}
    mean = roles['train'].mean(0)
    scale = roles['train'].std(0, correction=0)
    return roles, {'mean': mean.tolist(), 'scale': scale.tolist(),
                   'rule': 'per-coordinate population std from train only; raw=z*scale+mean'}


def standardize(x, stats):
    return (x-x.new_tensor(stats['mean']))/x.new_tensor(stats['scale'])


def inverse_standardize(x, stats):
    return x*x.new_tensor(stats['scale'])+x.new_tensor(stats['mean'])


class ExactPredictor:
    def __init__(self, distribution, stats, schedule):
        self.distribution, self.stats, self.schedule = distribution, stats, schedule

    def log_prob(self, x, t):
        from src.diffusion.schedules import extract
        a = extract(self.schedule.alpha_bar, t, x)
        centers = standardize(self.distribution.centers.to(x), self.stats)
        variance0 = (x.new_tensor(self.distribution.rho)/x.new_tensor(self.stats['scale'])).square()
        variance = a*variance0+(1-a)
        delta = x[:, None, :]-a.sqrt()[:, None, :]*centers[None, :, :]
        log_components = -.5*(delta.square()/variance[:, None, :]+variance[:, None, :].log()+math.log(2*math.pi)).sum(-1)
        return torch.logsumexp(log_components+self.distribution.weights.to(x).log(), 1)

    def prob(self, x, t):
        return self.log_prob(x, t).exp()

    def score(self, x, t):
        with torch.enable_grad():
            v = x.detach().requires_grad_(True)
            return torch.autograd.grad(self.log_prob(v, t).sum(), v)[0].detach()

    def __call__(self, x, t):
        from src.diffusion.schedules import extract
        return -(1-extract(self.schedule.alpha_bar, t, x)).sqrt()*self.score(x, t)


def datasets(cfg):
    roles, stats = roles_and_stats(cfg)
    hashes = {k: hashlib.sha256(x.numpy().tobytes()).hexdigest() for k, x in roles.items()}
    ds = {k: TensorDataset(standardize(v, stats).float(), torch.zeros(len(v), dtype=torch.long))
          for k, v in roles.items() if k != 'test'}
    if cfg['data'].get('test_train_size'):
        if cfg['experiment']['purpose'] != 'test':
            raise ValueError('Dataset truncation is unit-test-only')
        n = cfg['data']['test_train_size']
        ds['train'] = TensorDataset(*(v[:n] for v in ds['train'].tensors))
    return ds, {'dataset': 'rare_gmm2d', 'data_fingerprint': digest(hashes),
        'role_fingerprints': hashes, 'split_fingerprint': digest(cfg['data']),
        'split_sizes': {'train': len(ds['train']), 'val': 5000, 'test': 5000, 'cal': 0},
        'normalization': stats, 'data_role_protocol': 'train optimization; fixed val diagnostics; test excluded from training loaders; cal empty'}


def loaders(cfg, ds, generator):
    return {'train': DataLoader(ds['train'], batch_size=cfg['train']['batch_size'], shuffle=True,
                                drop_last=True, generator=generator, num_workers=0),
            'val': DataLoader(ds['val'], batch_size=1024, shuffle=False,
                              generator=torch.Generator().manual_seed(702), num_workers=0)}
