"""Measured resource budget, never a formal result."""
import json,time
from pathlib import Path
import torch,yaml
from src.data.multimodal_data import MelDataset,mel_collate
from src.models.audio_cnn import AudioCNN
from src.multimodal.fusion import FusionClassifier
from src.utils.reproducibility import seed_all
from src.utils.io import write_json

def main():
    seed_all(1,True,2);clips=[json.loads(p.read_text()) for p in Path('data/lab9/preprocessed').glob('*/metadata.json')];clips=[c for c in clips if c['role']=='train'][:16];assert len(clips)==16
    # Pilot normalization uses only these train clips; discarded pilot weights/stats.
    mel=torch.cat([torch.load(c['log_mel_path'],weights_only=True) for c in clips],1)
    stats={'mean':mel.mean(1).tolist(),'std':mel.std(1,unbiased=False).clamp_min(1e-6).tolist()}
    ds=MelDataset(clips,stats);x,y=mel_collate([ds[i] for i in range(16)]);x=x.cuda();y=y.cuda();m=AudioCNN({'classes':8}).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001);times=[]
    for i in range(25):
        torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad();loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward();opt.step();torch.cuda.synchronize()
        if i>=5:times.append(time.perf_counter()-t)
    import statistics
    step=statistics.median(times);cfg=yaml.safe_load(Path('configs/lab9.yaml').read_text());f=FusionClassifier(cfg['model']);xx=torch.randn(64,640);yy=torch.arange(64)%8;opt=torch.optim.AdamW(f.parameters(),lr=.001);t=time.perf_counter()
    for _ in range(100):opt.zero_grad();torch.nn.functional.cross_entropy(f(xx),yy).backward();opt.step()
    head=(time.perf_counter()-t)/100
    write_json('experiments/lab9/budget.json',{'purpose':'budget only; weights discarded; not formal results','genuine_train_clip_ids':[c['clip_id'] for c in clips],'audio_batch':16,'audio_step_seconds_median':step,'audio_100_epoch_upper_train_seconds':step*53*100,'audio_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'head_step_seconds':head,'head_200_epoch_train_seconds':head*14*200,'device':torch.cuda.get_device_name(),'threads':2,'audio_updates_pilot':25,'head_updates_pilot':100,'cal_read':False})
    print('BUDGET audio seconds/step',step,'head',head,flush=True)

if __name__=='__main__':main()
