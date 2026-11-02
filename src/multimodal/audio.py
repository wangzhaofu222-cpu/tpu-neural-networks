import torch


def mel_filterbank(fs=16000,n_fft=400,n_mel=64):
    # HTK mel scale and unnormalized triangular power filters, explicit protocol.
    lo=0.;hi=2595*torch.log10(torch.tensor(1+fs/2/700,dtype=torch.float64));m=torch.linspace(lo,float(hi),n_mel+2,dtype=torch.float64)
    hz=700*(10**(m/2595)-1);freq=torch.linspace(0,fs/2,n_fft//2+1,dtype=torch.float64)
    up=(freq[None,:]-hz[:-2,None])/(hz[1:-1]-hz[:-2])[:,None];down=(hz[2:,None]-freq[None,:])/(hz[2:]-hz[1:-1])[:,None]
    return torch.minimum(up,down).clamp_min(0)


def log_mel(waveform,fs=16000,n_mel=64,n_fft=400,hop_length=160,epsilon=1e-10):
    x=torch.as_tensor(waveform,dtype=torch.float64)
    if x.ndim!=1 or len(x)<n_fft//2+1:raise ValueError('Need valid mono audio long enough for reflect center padding')
    s=torch.stft(x,n_fft=n_fft,hop_length=hop_length,win_length=n_fft,window=torch.hann_window(n_fft,dtype=x.dtype),center=True,pad_mode='reflect',return_complex=True)
    return (mel_filterbank(fs,n_fft,n_mel)@s.abs().square()+epsilon).log().float()
