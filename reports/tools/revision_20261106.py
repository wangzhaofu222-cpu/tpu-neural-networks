"""Supplementary experiments. Reuses the shared runner; leaves old evidence intact."""
import argparse, copy, colorsys, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
import yaml
from torchvision.datasets import CIFAR10
from src.config import load_config
from src.engine.runner import ExperimentRunner
from src.engine.checkpoint import load_checkpoint
from src.models.small_cnn import build_model
from src.evaluation.features import extract_arrays, state_digest
from src.evaluation.ssl_evaluation import probe_config
from src.utils.io import write_json, file_hash, digest
from src.utils.provenance import create_run, git_state
from src.utils.reproducibility import seed_all

ROOT=Path('experiments/revision_20261106')
HUE=ROOT/'lab4'
SOURCES={'with_jitter':Path('runs/lab4/cf2097134c33eef5506c-be1a9df9a9714bfe9b95948d626f55c9'),
         'without_jitter':Path('runs/lab4/9bcb3e57a83e8f0b7c03-eba9695f16f344f7a05c93b54baaf23f')}

def read(p):return json.loads(Path(p).read_text('utf-8'))

def hue_assignments(n,seed):
    if n%10:raise ValueError('Balanced ten-class role size required')
    rng=np.random.default_rng(seed)
    labels=rng.permutation(np.arange(n,dtype=np.int64)%10)
    hues=(labels+.5)/10+rng.uniform(-.015,.015,n)
    return labels,hues

def colour_images(images,hues):
    x=images.astype(np.float32)/255
    v=.25+.65*(x[...,0]*.299+x[...,1]*.587+x[...,2]*.114)
    hv=np.asarray(hues)*6;sector=np.floor(hv).astype(int)%6;f=hv-np.floor(hv)
    p=np.full_like(hv,.2);q=1-.8*f;t=1-.8*(1-f)
    choices=np.stack([np.stack([np.ones_like(hv),t,p],-1),np.stack([q,np.ones_like(hv),p],-1),
                      np.stack([p,np.ones_like(hv),t],-1),np.stack([p,q,np.ones_like(hv)],-1),
                      np.stack([t,p,np.ones_like(hv)],-1),np.stack([np.ones_like(hv),p,q],-1)])
    rgb=choices[sector,np.arange(len(hues))]
    return np.rint(v[...,None]*rgb[:,None,None,:]*255).clip(0,255).astype(np.uint8)

def archive_controller(path):
    source=Path(__file__)
    (path/'controller_snapshot.py').write_bytes(source.read_bytes())
    return {'controller_sha256':file_hash(source),'protocol_sha256':file_hash(ROOT/'protocol.md')}

def feature_run(name,source):
    done=HUE/f'{name}_features.json'
    if done.exists():return Path(read(done)['path'])
    cfg=load_config('configs/lab4.yaml');seed_all(1,True,2)
    source_cfg=yaml.safe_load((source/'config.resolved.yaml').read_text('utf-8'))
    cp=source/'checkpoints/last.pt';cp_sha=file_hash(cp)
    model=build_model(source_cfg['model']).cuda().eval().requires_grad_(False)
    state=load_checkpoint(cp,model);before=state_digest(model)
    split=read('splits/cifar10.json');normalization=read(source/'manifest.json')['normalization']
    cfg['experiment'].update(purpose='formal',role='feature_extraction',name='Lab4_Q2_hue_'+name)
    cfg['logging']['run_root']='runs/revision_20261106/lab4'
    cfg['hue_control']={'protocol_sha256':file_hash(ROOT/'protocol.md'),'version':1,'role_seeds':{'train':4601,'val':4602,'test':4603},
                        'encoder_RUN_ID':source.name,'encoder_checkpoint_sha256':cp_sha,'class_rule':'hue only; semantic labels discarded'}
    fingerprint=digest(cfg['hue_control']);info={'data_fingerprint':fingerprint,'split_fingerprint':digest(split),
        'dataset':'HueCIFAR-10 constructed colour control','synthetic_fixture':False,'cal_access':False,'normalization':normalization}
    path,manifest=create_run(cfg,git_state(),info,model,source.name);controller=archive_controller(path)
    cache={'encoder_RUN_ID':source.name,'checkpoint':'last.pt','fingerprint':fingerprint,'split_fingerprint':digest(split)}
    hashes={};timing={};start=time.perf_counter()
    raw_train=CIFAR10('data',train=True,download=False).data
    raw_test=CIFAR10('data',train=False,download=False).data
    for role,seed in [('train',4601),('val',4602),('test',4603)]:
        ids=split['roles'][role];raw=raw_test if role=='test' else raw_train
        labels,hues=hue_assignments(len(ids),seed)
        # Only indexed train/val/test pixels are transformed. Cal is never indexed.
        images=colour_images(raw[ids],hues)
        hashes[role]=hashlib.sha256(images.tobytes()).hexdigest()
        np.savez_compressed(path/f'{role}_assignments.npz',indices=ids,labels=labels,hues=hues)
        then=time.perf_counter()
        h,z=extract_arrays(model,images,list(range(len(images))),normalization['mean'],normalization['std'],'cuda',256)
        timing[role]=time.perf_counter()-then
        if role=='val':
            hh,zz=extract_arrays(model,images,list(range(256)),normalization['mean'],normalization['std'],'cuda',256)
            assert torch.equal(h[:256],hh) and torch.equal(z[:256],zz)
        cache[role]={'h':h,'z_hat':z,'labels':torch.from_numpy(labels),'indices':ids,'role':role,'encoder_state_sha256':before}
        assert not np.intersect1d(ids,split['roles']['cal']).size if role!='test' else True
        assert np.array_equal(np.bincount(labels,minlength=10),np.full(10,len(ids)//10))
        # Save a deterministic visual audit gallery, independent of model predictions.
        if role=='train':
            gallery=np.stack([images[np.flatnonzero(labels==c)[0]] for c in range(10)])
            np.save(path/'class_examples.npy',gallery)
        print('FEATURES',name,role,len(ids),round(timing[role],2),flush=True)
    assert state_digest(model)==before and file_hash(cp)==cp_sha
    assert not set(split['roles']['train'])&set(split['roles']['val'])
    torch.save(cache,path/'features.pt')
    result={'RUN_ID':path.name,'path':path.as_posix(),'status':'COMPLETED_FEATURE_EXTRACTION',
            'encoder_RUN_ID':source.name,'encoder_checkpoint_sha256':cp_sha,'encoder_state_sha256':before,
            'feature_cache_sha256':file_hash(path/'features.pt'),'input_hashes':hashes,'cal_access':False,
            'role_sizes':{role:len(cache[role]['labels']) for role in ['train','val','test']},
            'seconds':time.perf_counter()-start,'role_seconds':timing,**controller}
    write_json(path/'summary.json',result);manifest.update(status=result['status'],global_step=0,**controller)
    write_json(path/'manifest.json',manifest);write_json(done,result)
    return path

def hue_probe(name,feature_path):
    result_path=HUE/f'{name}_probe.json'
    if result_path.exists():return read(result_path)
    cfg=load_config('configs/lab4.yaml');cfg['experiment']['purpose']='formal'
    cache=torch.load(feature_path/'features.pt',map_location='cpu',weights_only=False)
    candidates=[];started=time.perf_counter()
    # Train/val fitting and model choice finish before test prediction is computed.
    for regularization in [0.,.001,.01]:
        saved=HUE/f'{name}_candidate_{regularization:g}.json'
        if saved.exists():candidates.append(read(saved));continue
        pc=probe_config(cfg,feature_path/'features.pt','h',1,1.,cache['train']['indices'],regularization,'runs/revision_20261106/lab4')
        pc['experiment']['name']='Lab4_Q2_hue_'+name+'_L2_'+str(regularization)
        pc['hue_control']={'protocol_sha256':file_hash(ROOT/'protocol.md'),'feature_RUN_ID':feature_path.name}
        path=ExperimentRunner(pc).run();s=read(path/'summary.json')
        controller=archive_controller(path)
        manifest=read(path/'manifest.json');manifest.update(controller);write_json(path/'manifest.json',manifest)
        row={'regularization':regularization,'validation_accuracy':s['validation_metric'],'RUN_ID':path.name,'path':path.as_posix()}
        candidates.append(row);write_json(saved,row)
    winner=max(candidates,key=lambda r:(r['validation_accuracy'],-r['regularization']))
    chosen=Path(winner['path']);pc=yaml.safe_load((chosen/'config.resolved.yaml').read_text('utf-8'))
    model=build_model(pc['model']).cuda().eval();load_checkpoint(chosen/'checkpoints/last.pt',model)
    norm=read(chosen/'manifest.json')['normalization'];mean=torch.tensor(norm['feature_mean'],device='cuda');std=torch.tensor(norm['feature_std'],device='cuda')
    with torch.no_grad():
        probabilities=model((cache['test']['h'].cuda()-mean)/std).softmax(1).cpu()
    target=cache['test']['labels'];pred=probabilities.argmax(1)
    confusion=torch.bincount(target*10+pred,minlength=100).reshape(10,10)
    info={'data_fingerprint':cache['fingerprint'],'split_fingerprint':cache['split_fingerprint'],'synthetic_fixture':False,
          'dataset':'HueCIFAR-10 constructed colour control','chosen_probe_RUN_ID':chosen.name,'cal_access':False}
    ec=copy.deepcopy(pc);ec['experiment']['role']='probe_evaluation'
    path,m=create_run(ec,git_state(),info,model,cache['encoder_RUN_ID']);controller=archive_controller(path)
    result={'RUN_ID':path.name,'path':path.as_posix(),'status':'COMPLETED_PROBE_EVALUATION','condition':name,
            'feature_RUN_ID':feature_path.name,'encoder_RUN_ID':cache['encoder_RUN_ID'],'chosen_probe_RUN_ID':chosen.name,
            'validation_accuracy':winner['validation_accuracy'],'test_accuracy':(pred==target).double().mean().item(),
            'selected_regularization':winner['regularization'],'candidates':candidates,'confusion':confusion.tolist(),
            'test_N':len(target),'epochs_per_candidate':200,'cal_access':False,'seconds':time.perf_counter()-started,**controller}
    torch.save({'probabilities':probabilities,'labels':target,'predictions':pred},path/'predictions.pt')
    write_json(path/'summary.json',result);m.update(status=result['status'],global_step=0,**controller);write_json(path/'manifest.json',m)
    write_json(result_path,result);return result

def hue():
    HUE.mkdir(parents=True,exist_ok=True);start=time.perf_counter();outputs={}
    for name,source in SOURCES.items():outputs[name]=hue_probe(name,feature_run(name,source))
    assert read(HUE/'with_jitter_features.json')['input_hashes']==read(HUE/'without_jitter_features.json')['input_hashes']
    delta=outputs['without_jitter']['test_accuracy']-outputs['with_jitter']['test_accuracy']
    write_json(HUE/'results.json',{'conditions':outputs,'difference_without_minus_with':delta,
        'prediction_supported':delta>0,'protocol_sha256':file_hash(ROOT/'protocol.md'),'execution_seconds':time.perf_counter()-start,
        'scope':'one-seed control; not a multi-seed statistical claim'})
    print('Q2 COMPLETE',delta,flush=True)

def lab1_audit():
    old={r['label']:r for r in read('experiments/lab1/ledger.json')}
    new=read(ROOT/'lab1/ledger.json');rows=[]
    for r in new:
        label=r['label'];a=Path(old[label]['path']);b=Path(r['path']);x=read(a/'probe.initial.json');y=read(b/'probe.initial.json')
        s=read(b/'summary.json');initial_error=max(abs(x['activations'][k]['q']-y['activations'][k]['q'])/max(1e-30,abs(x['activations'][k]['q'])) for k in x['activations'])
        row={'label':label,'original_RUN_ID':a.name,'repeat_RUN_ID':b.name,'status':s['status'],'initial_q_max_relative_difference':initial_error,
             'trainability':s.get('trainability'),'global_step':s.get('global_step',read(b/'manifest.json')['global_step']),'validation_metric':s.get('validation_metric')}
        if 'c' in y:
            exact=math.prod(v['predicted_local_ratio_with_cross'] for v in y['branch_moments'].values())
            alpha=1/math.sqrt(len(y['branch_moments'])) if 'scaled' in label else 1.
            cross_free=math.prod(1+alpha*alpha*v['c'] for v in y['branch_moments'].values())
            bound=1.2*math.exp(y['c']);q=y['q_ratio']
            row.update(c=y['c'],q_ratio=q,scaled_bound=bound,exact_cross_term_product=exact,cross_free_layerwise_product=cross_free,
                       exact_identity_relative_error=abs(exact/q-1),threshold_pass=(q<=bound) if 'scaled' in label else abs(y['growth_per_block']/(1+y['c'])-1)<=.2)
            assert abs(exact/q-1)<1e-12
        assert initial_error<1e-12,(label,initial_error)
        rows.append(row)
    assert len(rows)==12
    result={'rows':rows,'implementation_error_found':False,'initial_moments_reproduced':True,
            'scaled_failures':[r['label'] for r in rows if 'scaled' in r['label'] and not r['threshold_pass']],
            'explanation':'Finite-width skip/branch cross moments are nonzero. Exact second-moment recurrence is verified; first-block c exponential approximation is not a deterministic bound on each finite network.'}
    write_json(ROOT/'lab1/results.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['hue','lab1_audit']);a=p.parse_args()
    hue() if a.stage=='hue' else lab1_audit()
