import copy,json,time
from pathlib import Path
import numpy as np
import torch,yaml
from src.engine.runner import ExperimentRunner
from src.models.small_cnn import build_model
from src.multimodal.fusion import late_probabilities
from src.utils.io import write_json,file_hash,digest
from src.utils.provenance import create_run,git_state
from src.utils.reproducibility import seed_all
from src.diagnostics.contribution import classification,majority_baseline,contribution

R=Path('experiments/lab9');D=Path('data/lab9');FEATURE=D/'features/pooled.pt'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def config():return yaml.safe_load(Path('configs/lab9.yaml').read_text(encoding='utf-8'))
def register(kind,path,**extra):
    registry=read(R/'run_registry.json') if (R/'run_registry.json').exists() else []
    registry.append({'kind':kind,'RUN_ID':Path(path).name,'path':str(path),**extra});write_json(R/'run_registry.json',registry)

def train(name,strategy,seed=1,width=128,p_drop=0.,purpose='formal',audio=False):
    cfg=config();cfg['experiment'].update(name=name,seed=seed,purpose=purpose)
    cfg['data']['normalization_path']=str(R/'mel_normalization.json')
    if audio:
        cfg['data']['name']='ravdess_audio';cfg['model']={'name':'audio_cnn','classes':8};cfg['runtime']['device']='cuda';cfg['train'].update(epochs=100,batch_size=16,early_stopping_patience=15)
    else:
        cfg['data']['feature_path']=str(FEATURE);cfg['model'].update(strategy=strategy,hidden_width=width,p_drop=p_drop)
    path=ExperimentRunner(cfg).run();register(name,path,strategy=strategy,seed=seed,purpose=purpose)
    return str(path)

def load(path):
    cfg=yaml.safe_load((Path(path)/'config.resolved.yaml').read_text(encoding='utf-8'));m=build_model(cfg['model'])
    state=torch.load(Path(path)/'checkpoints/best.pt',map_location='cpu',weights_only=False);m.load_state_dict(state['model']);m.eval().requires_grad_(False)
    return m,cfg,state

def features():return torch.load(FEATURE,weights_only=False)

@torch.no_grad()
def predict(spec,x,mask=(1,1)):
    if spec.get('strategy')=='late':
        pv,_=predict({'path':spec['video_path']},x,mask);pa,_=predict({'path':spec['audio_path']},x,mask)
        return late_probabilities(pv,pa,spec['weight'],mask),None
    m,_,_=load(spec['path']);v=x[:,:512];a=x[:,512:640];available=x[:,640:].bool()
    requested=torch.tensor(mask,dtype=torch.bool).expand(len(x),2)&available
    if (requested.sum(1)==0).any():raise ValueError('Requested only unavailable modality')
    p=m(visual=v,audio=a,mask=requested)
    return p,m.last_gate

def new_evaluation(name,spec,data,seed=1):
    cfg=config();cfg['experiment'].update(name=name,seed=seed,purpose='formal');code=git_state()
    if code['dirty']:raise RuntimeError('Commit source before formal evaluation')
    model,source_cfg,_=load(spec.get('path',spec.get('video_path')));cfg['model']=source_cfg['model'];cfg['evaluation_spec']=spec
    manifest=read(R/'data_manifest.json');info={'dataset':'RAVDESS official speech; frozen encoders','data_fingerprint':digest(data),'split_fingerprint':manifest['split_fingerprint'],'cal_read':False,'teacher_split_status':'TEACHER_SPLIT_UNVERIFIED'}
    path,passport=create_run(cfg,code,info,model);passport.update(status='COMPLETED_EVALUATION',global_step=0);write_json(path/'manifest.json',passport);register(name,path,seed=seed,purpose='formal',evaluation=True)
    return path

def evaluate(name,spec,role='test',x=None,seed=1,intervene=True):
    ds=features();d=ds[role];x=d['x'] if x is None else x;labels=d['y'];p,g=predict(spec,x)
    path=new_evaluation(name,spec,{'spec':spec,'role':role,'ids':d['clip_ids'],'input_sha256':digest(x.tolist())},seed)
    raw={'probabilities':p,'gates':g,'labels':labels,'clip_ids':d['clip_ids'],'spec':spec,'role':role};out={'RUN_ID':path.name,'spec':spec,'role':role,**classification(p.numpy(),labels.numpy())}
    if intervene:
        pv,gv=predict(spec,x,(1,0));pa,ga=predict(spec,x,(0,1));M0=majority_baseline(ds['train']['y'].numpy(),labels.numpy())['macro_f1']
        raw.update(no_audio=pv,no_video=pa,no_audio_gates=gv,no_video_gates=ga)
        out['contribution']=contribution(p.numpy(),pv.numpy(),pa.numpy(),labels.numpy(),M0)
        if g is not None:
            from src.diagnostics.gate_stats import gate_statistics
            out['gate']=gate_statistics(g.numpy(),labels.numpy(),(p.argmax(1)==labels).numpy())
    raw_path=path/'predictions.pt';torch.save(raw,raw_path);out['predictions_sha256']=file_hash(raw_path)
    out['fusion_profile']={'parameters':read(Path(spec.get('path',spec.get('video_path')))/'manifest.json')['trainable_parameters'],'macro_f1':out['macro_f1'],'contribution':out.get('contribution'),'undefined':out.get('contribution',{}).get('undefined',[]),'pair_following':'NOT_APPLICABLE: original labelled clips'}
    write_json(path/'results.json',out);write_json(path/'summary.json',out);return out

def matched_control(target,strategy):
    from src.multimodal.fusion import matching_width
    return matching_width(512 if strategy=='video' else 128,8,target)

def training_capacity(model):
    """Count the trained architecture, independently of inference requires_grad flags."""
    total=sum(p.numel() for p in model.parameters());head=sum(p.numel() for p in model.head.parameters())
    return {'frozen_encoders':0,'fusion_trainable':total-head,'head_trainable':head,'total_trainable':total,'encoders_external_frozen_cache':True,'count_basis':'architecture trained in source run; inference freeze flags do not erase its capacity'}

def exploratory_train(name,strategy,width=128):
    # A bookkeeping failure must not force needless retraining of unchanged real runs.
    for row in reversed(read(R/'run_registry.json') if (R/'run_registry.json').exists() else []):
        if row['kind']!=name:continue
        path=Path(row['path']);cfg=yaml.safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8'));passport=read(path/'manifest.json')
        assert passport['status']=='COMPLETED' and passport['global_step']==2800
        assert cfg['model']['strategy']==strategy and cfg['model']['hidden_width']==width and cfg['model']['p_drop']==0
        assert passport['seed']==1 and passport['data_fingerprint']==file_hash(FEATURE)
        return str(path)
    return train(name,strategy,width=width)
