"""Frozen-model evaluation uses the Lab0 passport/EXP_ID/RUN_ID mechanisms."""
import copy,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch,yaml
from torchvision.datasets import CIFAR10
from src.models.small_cnn import build_model
from src.attacks import NormalizedModel,fgsm,pgd
from src.corruptions.image import corrupt_image
from src.reliability.metrics import classification_metrics
from src.reliability.uncertainty import uncertainty
from src.utils.io import write_json,digest,file_hash
from src.utils.provenance import create_run,git_state
from src.utils.reproducibility import seed_all

R=Path('experiments/lab8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def config():return yaml.safe_load(Path('configs/lab8.yaml').read_text(encoding='utf-8'))
def registry():return read(R/'run_index.json') if (R/'run_index.json').exists() else {}
def register(key,path):
    idx=registry();idx[key]=str(path);write_json(R/'run_index.json',idx)

_data={}
def role_data(role):
    if role not in ['val','test','ood']:raise ValueError('Lab8 never reads cal')
    if role in _data:return _data[role]
    split=read('splits/cifar10.json')
    if role=='ood':
        import scipy.io
        p=Path('data/svhn/test_32x32.mat')
        from torchvision.datasets import SVHN
        assert hashlib.md5(p.read_bytes()).hexdigest()==SVHN.split_list['test'][2]
        # Only X is requested: semantic digit labels are never loaded into analysis.
        images=scipy.io.loadmat(p,variable_names=['X'])['X'].transpose(3,2,0,1)
        x=torch.from_numpy(images.copy());labels=None;ids=[f'svhn-test:{i}' for i in range(len(x))]
    else:
        ds=CIFAR10('data',train=role=='val',download=False);indices=split['roles'][role]
        x=torch.from_numpy(ds.data[indices].transpose(0,3,1,2).copy());labels=torch.tensor(np.array(ds.targets)[indices]);ids=[f'cifar10-{role}:{i}' for i in indices]
    _data[role]=(x,labels,ids);return _data[role]

def source_model(key,device='cuda'):
    row=read(R/'model_sources.json')[key];p=Path(row['path']);assert file_hash(p)==row['sha256']
    state=torch.load(p,map_location='cpu',weights_only=False);m=build_model(row['cfg']['model']);m.load_state_dict(state['model'],strict=True)
    manifest=read(p.parent.parent/'manifest.json');norm=manifest['normalization'];split=read('splits/cifar10.json')
    assert manifest['split_fingerprint']==split['fingerprint'] and state['run_id']==row['RUN_ID']
    m=NormalizedModel(m,norm['mean'],norm['std']).to(device).eval()
    for parameter in m.parameters():parameter.requires_grad_(False)
    return m,row

def observation(key,details,model,source,*,purpose='formal'):
    idx=registry()
    if key in idx:return Path(idx[key]),read(Path(idx[key])/'manifest.json')
    cfg=copy.deepcopy(config());cfg['experiment'].update(role=key,purpose=purpose,seed=source['seed']);cfg['evaluation']=details
    code=git_state()
    if purpose=='formal' and code['dirty']:raise RuntimeError('Commit source/config/tests before formal evaluation')
    split=read('splits/cifar10.json')
    info={'data_fingerprint':source['sha256'],'split_fingerprint':split['fingerprint'],'checkpoint_sha256':source['sha256'],'source_RUN_ID':source['RUN_ID'],
          'data_roles':{'temperature_fit':'val','evaluation':'test or SVHN test','cal':'not read'},'course_protocol_status':'UNVERIFIED','normalization':read(Path(source['path']).parent.parent/'manifest.json')['normalization']}
    p,manifest=create_run(cfg,code,info,model,source['RUN_ID']);register(key,p);return p,manifest

def finish(p,manifest,summary):
    write_json(p/'summary.json',summary);manifest.update(status='COMPLETED_EVALUATION',completed_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,wall_seconds=summary.get('wall_seconds',0));write_json(p/'manifest.json',manifest)

def records(path,z,y,ids,temperature,metadata):
    p=(z.double()/temperature).softmax(1);pred=z.argmax(1);u=uncertainty(z,temperature)
    with Path(path).open('w',encoding='utf-8') as f:
        for i,obj in enumerate(ids):
            row={**metadata,'object_ID':obj,'logits':z[i].tolist(),'prediction':int(pred[i]),'confidence':float(p[i].max()),
                 'U_msp':float(u['msp'][i]),'H':float(u['entropy'][i]),'U_margin':float(u['margin'][i]),'correct':bool(pred[i]==y[i]) if y is not None else None,'OOD':y is None,'temperature':temperature}
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def evaluate(key,source_key,role='test',kind='clean',level=0,attack_cfg=None,*,purpose='formal',limit=None,interrupt_after_batches=None):
    idx=registry()
    if key in idx and (Path(idx[key])/'summary.json').exists():return Path(idx[key])
    cfg=config();seed_all(1,True,2);device=cfg['runtime']['device'];m,source=source_model(source_key,device)
    x,y,ids=role_data(role)
    if limit is not None:x=x[:limit];y=y[:limit] if y is not None else None;ids=ids[:limit]
    details={'source_key':source_key,'role':role,'corruption':kind,'level':level,'attack':attack_cfg,'object_IDs_hash':digest(ids),'N':len(ids)}
    p,manifest=observation(key,details,m,source,purpose=purpose);parts=p/'batches';parts.mkdir(exist_ok=True);start=time.perf_counter();allz=[];constraints=[];batch_size=cfg['attacks']['batch_size']
    state_before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    for b in range(0,len(x),batch_size):
        f=parts/f'{b:06}.pt'
        if f.exists():out=torch.load(f,weights_only=False)
        else:
            raw=x[b:b+batch_size].float()/255;labels=y[b:b+batch_size].to(device) if y is not None else None
            if kind!='clean':raw=torch.stack([corrupt_image(a,kind,level,1,i) for a,i in zip(raw,ids[b:b+batch_size])])
            raw=raw.to(device);info=None
            if attack_cfg:
                a=attack_cfg
                if a['name']=='fgsm':raw,info=fgsm(m,raw,labels,a['epsilon'])
                elif a['name']=='pgd':
                    # Explicit math backend supports reliable input gradients on ViT.
                    from torch.nn.attention import sdpa_kernel,SDPBackend
                    with sdpa_kernel(SDPBackend.MATH):raw,info=pgd(m,raw,labels,a['epsilon'],K=a['K'],R=a['R'],seed=100000+b)
                elif a['name']=='random':
                    g=torch.Generator(device=device).manual_seed(100000+b);clean=raw;raw=(raw+a['epsilon']*torch.randn(raw.shape,generator=g,device=device).sign()).clamp(0,1)
                    from src.attacks.fgsm import constraint
                    info={'constraints':[constraint(clean,raw,a['epsilon'])]}
                else:raise ValueError('Unknown attack')
            with torch.no_grad():z=m(raw).detach().cpu()
            out={'logits':z,'object_IDs':ids[b:b+batch_size],'constraints':info['constraints'] if info else []}
            if attack_cfg:out['adversarial_pixels']=raw.cpu();out['attack_info']={k:v for k,v in info.items() if k!='constraints'}
            torch.save(out,f)
        assert out['object_IDs']==ids[b:b+batch_size];allz.append(out['logits']);constraints.extend(out['constraints'])
        if b%1024==0:print(key,b,'/',len(x),flush=True)
        if interrupt_after_batches and (b//batch_size+1)==interrupt_after_batches:
            if purpose!='smoke':raise ValueError('Fault injection allowed only for smoke')
            manifest['status']='INTERRUPTED_SMOKE';write_json(p/'manifest.json',manifest)
            raise InterruptedError('Intentional smoke interruption after durable shards')
    z=torch.cat(allz);torch.cuda.synchronize() if device=='cuda' else None
    assert not m.training and all(torch.equal(state_before[k],v.detach().cpu()) for k,v in m.state_dict().items())
    elapsed=time.perf_counter()-start
    torch.save({'logits':z,'labels':y,'object_IDs':ids,'role':role,'OOD':role=='ood'},p/'outputs.pt')
    metric=None if y is None else classification_metrics(z,y)
    summary={'RUN_ID':p.name,'source_key':source_key,'source_RUN_ID':source['RUN_ID'],'checkpoint_sha256':source['sha256'],'N':len(x),'role':role,'metrics':metric,'wall_seconds':elapsed,'constraints':constraints,'parameters_unchanged':True,'model_eval':not m.training,'optimizer_updates':0,**details}
    records(p/'records_T1.jsonl',z,y,ids,1.,{'RUN_ID':p.name,'model':source_key,'seed':source['seed'],'corruption':kind,'level':level})
    finish(p,manifest,summary);return p

def outputs(p):return torch.load(Path(p)/'outputs.pt',map_location='cpu',weights_only=False)

def pair_statistics(deltas):
    d=np.asarray(deltas,dtype=float);mean=float(d.mean());sd=float(d.std(ddof=1));same_positive=bool((d>0).all());same_negative=bool((d<0).all())
    established=(same_positive or same_negative) and abs(mean)>=2*sd
    return {'deltas':d.tolist(),'mean':mean,'sample_sd':sd,'all_positive':same_positive,'all_negative':same_negative,'abs_mean_ge_2sd':abs(mean)>=2*sd,
            'conclusion':'на чистых данных температура помогает сильнее' if established and same_positive else ('обратный вывод установлен' if established else 'различие переноса не установлено'),'expected_positive_supported':bool(established and same_positive)}
