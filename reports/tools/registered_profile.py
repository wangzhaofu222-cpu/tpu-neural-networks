"""Real E4 measurements with their own immutable manifest and RUN_ID."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import torch
from src.config import load_config
from src.models.small_cnn import build_model
from src.engine.checkpoint import load_checkpoint
from src.profile import profile_model
from src.utils.io import file_hash,digest,write_json
from src.utils.provenance import create_run,git_state
from src.utils.reproducibility import seed_all


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args(); cfg=load_config(a.run/'config.resolved.yaml')
    cfg['experiment'].update(name='lab0_E4_'+a.device,purpose='formal_profiling',diagnostic_script_sha256=file_hash(__file__))
    cfg['runtime']['device']=a.device; cfg['profiling'].update(warmup=50,repeats=100)
    device=torch.device(a.device)
    seed_all(cfg['experiment']['seed'],True,2); model=build_model(cfg['model']).to(device)
    checkpoint=a.run/'checkpoints/last.pt'; load_checkpoint(checkpoint,model)
    source=json.loads((a.run/'manifest.json').read_text())
    info={'dataset':'explicit normal profiling tensors, Bx3x32x32; no accuracy measurement',
        'data_fingerprint':digest({'checkpoint':file_hash(checkpoint),'seed':cfg['experiment']['seed']}),
        'split_fingerprint':source['split_fingerprint'],'synthetic_fixture':False}
    path,manifest=create_run(cfg,git_state(ROOT),info,model,parent_run_id=a.run.name)
    result=profile_model(model,cfg['profile_batch_sizes'],device,50,100)
    result.update(RUN_ID=path.name,source_RUN_ID=a.run.name,checkpoint_sha256=file_hash(checkpoint),status='REAL_E4_MEASUREMENTS; CV flags retained')
    write_json(path/'profile.json',result); write_json(path/'summary.json',result)
    (path/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result['measurements']),encoding='utf-8')
    manifest['status']='COMPLETED_PROFILING_ONLY'; write_json(path/'manifest.json',manifest); write_json(a.out,result)
    print(path)


if __name__=='__main__': main()
