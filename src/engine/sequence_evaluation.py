"""Sequence evaluation/profiling operations use the shared provenance and RNG harness."""
import copy
import time
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from src.data.synthetic_sequence import Sequences
from src.models.small_cnn import build_model
from src.metrics.classification import evaluate_loader
from src.engine.checkpoint import load_checkpoint
from src.utils.io import digest,file_hash,write_json
from src.utils.provenance import create_run,git_state
from src.utils.reproducibility import seed_all

def evaluate(source,length,rule='P2',role='test',diagnostics=False):
    if role not in {'val','test'}:raise PermissionError('Evaluation supports only val/test; cal reserved')
    source=Path(source);cfg=yaml.safe_load((source/'config.resolved.yaml').read_text(encoding='utf-8'))
    cfg=copy.deepcopy(cfg);cfg['experiment']['role']='evaluation'
    cfg['evaluation']={'length':length,'absolute_rule':rule,'role':role,'attention':diagnostics}
    cfg['model']['absolute_rule']=rule
    code=git_state();seed_all(cfg['experiment']['seed'],True,cfg['runtime']['threads'])
    model=build_model(cfg['model']).to(cfg['runtime']['device']);checkpoint=source/'checkpoints/last.pt'
    state=load_checkpoint(checkpoint,model)
    if state['global_step']!=cfg['train']['max_steps']:raise ValueError('Incomplete source checkpoint')
    ds=Sequences(cfg['data']['task'],cfg['data']['L0'],cfg['experiment']['seed'],role,cfg['data']['evaluation_examples'],32,length)
    input_sha=digest([[t.tolist() for t in ds.sample(i)] for i in range(len(ds))])
    info={'data_fingerprint':input_sha,'split_fingerprint':digest({'role':role,'seed':cfg['experiment']['seed'],'length':length}),
          'parent_checkpoint_sha256':file_hash(checkpoint),'parent_RUN_ID':source.name,'synthetic_fixture':False}
    path,m=create_run(cfg,code,info,model,source.name);start=time.perf_counter()
    try:
        result=evaluate_loader(model,DataLoader(ds,batch_size=4,shuffle=False),cfg['runtime']['device'])
        result.update(status='COMPLETED_EVALUATION',RUN_ID=path.name,parent_RUN_ID=source.name,length=length,
                      absolute_rule=rule,position=cfg['model']['position'],seed=cfg['experiment']['seed'],role=role,
                      input_sha256=input_sha,checkpoint_sha256=info['parent_checkpoint_sha256'],optimizer_steps=0)
        if diagnostics:
            from src.diagnostics.attention_stats import attention_statistics,summarize
            model.eval();raw=[];summaries=[]
            # All200 examples, four at a time: full per-query raw summaries without retaining L² matrices.
            accum=[{} for _ in model.blocks]
            with torch.no_grad():
                for lo in range(0,len(ds),4):
                    triples=[ds.sample(i) for i in range(lo,min(lo+4,len(ds)))];x=torch.stack([v[0] for v in triples]).to(cfg['runtime']['device'])
                    targets=torch.stack([v[2] for v in triples]).to(x.device)
                    _,matrices=model(x,return_attention=True)
                    for block,a in enumerate(matrices):
                        stats=attention_statistics(a,targets)
                        for name,v in stats.items():accum[block].setdefault(name,[]).append(v.cpu())
                for block,values in enumerate(accum):
                    values={k:torch.cat(v,0) for k,v in values.items()};raw.append(values)
                    summaries.append({'block':block,**summarize(values)})
            torch.save(raw,path/'attention_raw.pt');write_json(path/'attention.json',summaries)
            result['attention']=summaries;result['attention_backend']='explicit; same arithmetic as trained explicit model'
    except torch.OutOfMemoryError as exc:
        result={'status':'OOM','RUN_ID':path.name,'parent_RUN_ID':source.name,'length':length,'exception':str(exc),'optimizer_steps':0}
        torch.cuda.empty_cache()
    result['wall_seconds']=time.perf_counter()-start;write_json(path/'summary.json',result)
    m.update(status=result['status'],global_step=0,wall_seconds=result['wall_seconds']);write_json(path/'manifest.json',m)
    return path,result

def profile(cfg,length,dtype=torch.float32):
    from src.profiling.sequence_profiler import safe_measure
    code=git_state()
    if cfg['experiment']['purpose']=='formal' and code['dirty']:raise RuntimeError('Formal profile requires committed source')
    seed_all(cfg['experiment']['seed'],True,cfg['runtime']['threads'])
    model=build_model(cfg['model']).to(device=cfg['runtime']['device'],dtype=dtype)
    info={'data_fingerprint':digest({'length':length,'batch':4,'dtype':str(dtype),'seed':cfg['experiment']['seed']}),
          'split_fingerprint':digest({'profiling_only':True}),'synthetic_fixture':False}
    path,m=create_run(cfg,code,info,model)
    try:result=safe_measure(model,length,warmup=cfg['profiling']['warmup'],repeats=cfg['profiling']['repeats'])
    except Exception as exc:
        result={'status':'FAILED_PROFILE','RUN_ID':path.name,'length':length,'exception':str(exc)}
        write_json(path/'failure.json',result);write_json(path/'summary.json',result)
        m.update(status='FAILED_PROFILE',global_step=0);write_json(path/'manifest.json',m)
        raise
    result.update(RUN_ID=path.name,optimizer_steps=0,operation='resource_profiling_training_step_not_scientific_training')
    write_json(path/'summary.json',result);write_json(path/'resources.json',result)
    m.update(status=result['status'],global_step=0);write_json(path/'manifest.json',m)
    return path,result
