import torch
from torch import nn
from .time_embedding import SinusoidalEmbedding


class EpsilonMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        w, depth, e = cfg['width'], cfg['depth'], cfg['time_dim']
        self.embedding = SinusoidalEmbedding(e)
        self.time_mlp = nn.Sequential(nn.Linear(e, e), nn.SiLU(), nn.Linear(e, e))
        self.layers = nn.ModuleList(nn.Linear(2 if i == 0 else w, w) for i in range(depth))
        self.time_projections = nn.ModuleList(nn.Linear(e, w, bias=False) for _ in range(depth))
        self.output = nn.Linear(w, 2)

    def forward(self, x, t):
        t = torch.as_tensor(t, device=x.device)
        e = self.time_mlp(self.embedding(t).to(x.dtype))
        for layer, projection in zip(self.layers, self.time_projections):
            x = torch.nn.functional.silu(layer(x)+projection(e))
        return self.output(x)
