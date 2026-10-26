"""E4 profiler; measurements are real and independent at every batch size."""
import argparse
import json
import statistics
import time
import numpy as np
import torch
from src.config import load_config
from src.models.small_cnn import build_model
from src.utils.io import write_json
from src.utils.reproducibility import seed_all
from src.utils.provenance import create_run,git_state
from src.utils.io import digest


def profile_model(model,batch_sizes,device,warmup=10,repeats=30):
    if warmup<10 or repeats<30:
        raise ValueError('Lab 0 E4 requires at least 10 warmup and 30 measured calls')
    def sync():
        if device.type=='cuda':
            torch.cuda.synchronize(device)
    was_training = model.training
    model.eval()
    results = []
    try:
        for batch in batch_sizes:
            x = torch.randn(batch,3,32,32,device=device)
            with torch.no_grad():
                sync()
                start=time.perf_counter()
                model(x)
                sync()
                cold=time.perf_counter()-start
                for _ in range(warmup):
                    model(x)
                sync()
                if device.type=='cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                samples=[]
                for _ in range(repeats):
                    sync()
                    start=time.perf_counter()
                    model(x)
                    sync()
                    samples.append(time.perf_counter()-start)
            avg=statistics.mean(samples)
            cv=statistics.stdev(samples)/avg
            results.append({'batch_size':batch,'cold_latency_seconds':cold,'mean_seconds':avg,
                'median_seconds':statistics.median(samples),'p95_seconds':float(np.percentile(samples,95)),
                'cv':cv,'cv_acceptable':cv<=.1,'throughput_samples_per_second':batch/avg,
                'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                'samples_seconds':samples,'warmup':warmup,'repeats':repeats,'operation':'eval_forward_batch'})
    finally:
        model.train(was_training)
    return {'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device':str(device),'measurements':results,
            'note':'Forward-only batch latency; training epoch time comes from metrics.jsonl. CV failures require repeats or diagnosis.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--set',action='append',default=[])
    p.add_argument('--out',default='docs/environment/profile-development.json')
    a=p.parse_args()
    cfg=load_config(a.config,a.set)
    seed_all(cfg['experiment']['seed'],cfg['experiment']['deterministic'],cfg['runtime']['threads'])
    device=torch.device(cfg['runtime']['device'])
    model=build_model(cfg['model']).to(device)
    data_info={'dataset':'explicit profiling tensors; not training data',
               'data_fingerprint':digest({'shape':[3,32,32],'distribution':'normal','seed':cfg['experiment']['seed']}),
               'split_fingerprint':digest({'no_dataset_split_for_model_profiling':True}),
               'synthetic_fixture':False}
    path,manifest=create_run(cfg,git_state(),data_info,model)
    result=profile_model(model,cfg['profile_batch_sizes'],device,**cfg['profiling'])
    result['config']=cfg
    result['status']='DEVELOPMENT_MEASUREMENT_NOT_FORMAL_E4'
    result['RUN_ID']=manifest['RUN_ID']
    write_json(path/'profile.json',result)
    manifest['status']='COMPLETED_PROFILING_ONLY'
    write_json(path/'manifest.json',manifest)
    write_json(a.out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'config','measurements'}},indent=2))


if __name__=='__main__':
    main()
