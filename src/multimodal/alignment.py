from fractions import Fraction
import torch


def alignment_windows(video_indices,audio_length,*,fs=16000,video_fps=5,hop=160,r=1,av_offset_ms=0,audio_shift_ms=0,window_samples=None,valid_audio=None):
    """Half-open windows in integer sample units; offset shifts audio timestamps."""
    if fs%video_fps or min(fs,video_fps,hop,r)<=0:raise ValueError('Require integral video sample spacing')
    width=fs//video_fps if window_samples is None else window_samples
    if width!=fs//video_fps:raise ValueError('Mandatory nonoverlapping/no-gap protocol requires width=1/fps')
    offset=Fraction(str(av_offset_ms))+Fraction(str(audio_shift_ms));offset=offset*fs/1000
    if offset.denominator!=1:raise ValueError('Offset must resolve to integer audio samples')
    a=torch.arange(audio_length,dtype=torch.int64)*hop*r+int(offset)
    valid=torch.ones(audio_length,dtype=torch.bool) if valid_audio is None else torch.as_tensor(valid_audio,dtype=torch.bool)
    windows=[]
    for t in video_indices:
        center=int(t)*(fs//video_fps)
        # Doubled integer units also support odd window lengths without rounding.
        windows.append(torch.where(valid & (2*(a-center)>=-width) & (2*(a-center)<width))[0])
    assigned=torch.cat(windows) if windows else torch.empty(0,dtype=torch.long)
    if len(assigned)!=len(torch.unique(assigned)):raise ValueError('Overlapping windows')
    return {'windows':windows,'audio_sample_times':a,'assigned_indices':assigned,'offset_samples':int(offset),'window_samples':width,'boundary':'left inclusive, right exclusive'}


def masked_mean(x,valid,dim=-2):
    mask=torch.as_tensor(valid,device=x.device,dtype=torch.bool)
    count=mask.sum(-1,keepdim=True)
    if (count==0).any():raise ValueError('No valid frames; represent missing modality explicitly')
    return (x*mask.unsqueeze(-1)).sum(dim)/count.to(x.dtype)
