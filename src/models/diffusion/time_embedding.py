import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        if dimension % 2:
            raise ValueError('Even embedding dimension required')
        self.register_buffer('frequencies', 10000**(-2*torch.arange(dimension//2, dtype=torch.float64)/dimension))

    def forward(self, t):
        phase = t.reshape(-1, 1).to(self.frequencies)*self.frequencies
        return torch.cat([phase.sin(), phase.cos()], 1)
