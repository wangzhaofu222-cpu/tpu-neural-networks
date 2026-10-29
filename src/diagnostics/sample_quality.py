"""Original-unit Lab7 quality against a frozen100000-point mode reference."""
import torch
from src.diagnostics.density_metrics import kernel_mean


def frequencies(x, distribution):
    return torch.bincount(distribution.mode_id(x), minlength=distribution.M+1).double()/len(x)


def quality(samples, test, distribution, reference, h, test_kernel=None):
    samples, test = samples.cpu().double(), test.cpu().double()
    freq = frequencies(samples, distribution)
    reference = torch.as_tensor(reference, dtype=torch.float64)
    xx = kernel_mean(test, test, h) if test_kernel is None else test_kernel
    mmd = xx+kernel_mean(samples, samples, h)-2*kernel_mean(samples, test, h)
    return {'Cov': float((freq[:-1]>=reference[:-1]/2).double().mean()),
            'mode_frequencies': freq.tolist(), 'rare_frequency': float(freq[0]),
            'rare_covered_pdf': bool(freq[0]>=reference[0]/2),
            'rare_covered_pi0': bool(freq[0]>=distribution.pi0/2),
            'Dmode': float((freq-reference).abs().sum()/2), 'MMD2': float(mmd)}
