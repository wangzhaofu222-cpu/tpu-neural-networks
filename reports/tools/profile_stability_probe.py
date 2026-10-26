"""Localize timing instability by adding a measured one-second device settling warmup."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from src.config import load_config
from src.models.small_cnn import build_model
from src.engine.checkpoint import load_checkpoint
from src.profile import profile_model
from src.utils.io import file_hash,digest,write_json
from src.utils.provenance import create_run,git_state
from src.utils.reproducibility import seed_all


def clocks():
    r=subprocess.run(['nvidia-smi','--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else None


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); cfg=load_config(a.run/'config.resolved.yaml')
    cfg['runtime']['device']='cuda'; cfg['experiment'].update(purpose='formal_profiling',name='lab0_GPU_warmup_stability',diagnostic_script_sha256=file_hash(__file__))
    cfg['profiling'].update(warmup=50,repeats=300,settle_seconds=1.)
    seed_all(cfg['experiment']['seed'],True,2); device=torch.device('cuda')
    model=build_model(cfg['model']).to(device); load_checkpoint(a.run/'checkpoints/last.pt',model); model.eval()
    info={'dataset':'normal profiling tensors','data_fingerprint':digest({'checkpoint':file_hash(a.run/'checkpoints/last.pt'),'seed':cfg['experiment']['seed']}),
          'split_fingerprint':json.loads((a.run/'manifest.json').read_text())['split_fingerprint'],'synthetic_fixture':False}
    path,manifest=create_run(cfg,git_state(ROOT),info,model,parent_run_id=a.run.name); rows=[]
    for batch in cfg['profile_batch_sizes']:
        x=torch.randn(batch,3,32,32,device=device); before=clocks(); count=0; start=time.perf_counter()
        with torch.no_grad():
            while time.perf_counter()-start<1:
                model(x); torch.cuda.synchronize(device); count+=1
        settling_elapsed=time.perf_counter()-start
        after=clocks(); row=profile_model(model,[batch],device,50,300)['measurements'][0]
        row.update(settling_iterations=count,settling_seconds=settling_elapsed,clocks_before=before,clocks_after=after)
        rows.append(row)
    result={'RUN_ID':path.name,'source_RUN_ID':a.run.name,'measurements':rows,'status':'REAL_E4_GPU_REPEAT; raw CV flags retained',
            'clock_fields':'SM MHz, memory MHz, GPU Celsius, W, pstate','settling_is_additional_to_50_warmup_calls':True}
    write_json(path/'profile.json',result); write_json(path/'summary.json',result)
    (path/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    manifest['status']='COMPLETED_PROFILING_ONLY'; write_json(path/'manifest.json',manifest)
    write_json(a.out,result); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
