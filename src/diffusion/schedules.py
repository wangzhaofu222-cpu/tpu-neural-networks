from dataclasses import dataclass
import torch


@dataclass
class Schedule:
    beta: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor
    snr: torch.Tensor
    log_snr: torch.Tensor
    posterior_var: torch.Tensor
    kind: str
    T: int
    protocol_valid: bool

    def to(self, device):
        return Schedule(*(getattr(self, k).to(device) for k in
                          ['beta', 'alpha', 'alpha_bar', 'snr', 'log_snr', 'posterior_var']),
                        self.kind, self.T, self.protocol_valid)


def make_schedule(config, diagnostic=False):
    T = int(config['T'])
    if T < 2:
        raise ValueError('Schedule requires T>=2')
    beta = torch.zeros(T+1, dtype=torch.float64)
    kind = config['kind']
    if kind == 'linear':
        beta[1:] = torch.linspace(config['beta_min'], config['beta_max'], T, dtype=torch.float64)
    elif kind == 'cosine':
        t = torch.arange(T+1, dtype=torch.float64)
        f = torch.cos((t/T+.008)/1.008*torch.pi/2).square()
        beta[1:] = (1-f[1:]/f[:-1]).clamp(max=.999)
    else:
        raise ValueError('Unknown schedule')
    if not bool(((beta[1:] > 0) & (beta[1:] < 1)).all()):
        raise ValueError('Invalid beta')
    alpha = 1-beta
    ab = torch.cumprod(alpha, 0)
    snr = ab/(1-ab)
    posterior = torch.zeros_like(beta)
    posterior[1:] = beta[1:]*(1-ab[:-1])/(1-ab[1:])
    valid = float(ab[-1]) <= 1e-3
    if not valid and not diagnostic:
        raise ValueError('Terminal alpha_bar exceeds 1e-3; diagnostic must be explicit')
    return Schedule(beta, alpha, ab, snr, snr.log(), posterior, kind, T, valid)


def extract(values, t, x):
    v = values.to(x.device)[torch.as_tensor(t, device=x.device, dtype=torch.long)].to(x.dtype)
    return v.reshape((-1,)+(1,)*(x.ndim-1))


def terminal_kl(schedule, train):
    a = schedule.alpha_bar[-1].cpu()
    return float(.5*(a*train.double().square().sum(1)-train.shape[1]*a
                     -train.shape[1]*torch.log1p(-a)).mean())
