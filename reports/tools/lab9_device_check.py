"""Actual CPU/CUDA inference agreement on fixed validation inputs."""
import copy,torch
from reports.tools.lab9_common import R,read,load,features
from src.data.multimodal_data import MelDataset,mel_collate
from src.utils.reproducibility import seed_all
from src.utils.io import write_json

@torch.no_grad()
def main():
    seed_all(1,True,2)
    if not torch.cuda.is_available():write_json(R/'cpu_gpu_consistency.json',{'status':'NOT_APPLICABLE','reason':'CUDA unavailable'});return
    clips=[c for c in read(R/'data_manifest.json')['clips'] if c['role']=='val'][:4];ds=MelDataset(clips,read(R/'mel_normalization.json'));x,y=mel_collate([ds[i] for i in range(4)]);source=read(R/'encoder_sources.json')['audio'];m,_,_=load(source['run_path']);a=m(x);b=m.cuda()(x.cuda()).cpu();audio_error=float((a-b).abs().max());audio_ok=torch.allclose(a,b,atol=1e-4,rtol=1e-4)
    row=next(r for r in read(R/'run_registry.json') if r['kind']=='E1_explore_video');m,_,_=load(row['path']);x=features()['val']['x'][:16];a=m(x);b=m.cuda()(x.cuda()).cpu();head_error=float((a-b).abs().max());head_ok=torch.allclose(a,b,atol=1e-4,rtol=1e-4)
    out={'status':'PASS' if audio_ok and head_ok else 'FAIL','audio_max_absolute_logit_error':audio_error,'head_max_absolute_logit_error':head_error,'atol':1e-4,'rtol':1e-4,'audio_RUN_ID':source['RUN_ID'],'head_RUN_ID':row['RUN_ID'],'audio_clip_ids':[c['clip_id'] for c in clips],'role':'val','no_training':True,'weights_changed':False,'device':torch.cuda.get_device_name()};write_json(R/'cpu_gpu_consistency.json',out);print(out)

if __name__=='__main__':main()
