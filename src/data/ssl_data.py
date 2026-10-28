"""Lab4 roles reuse the immutable Lab0 split. SSL path never accesses targets."""
import hashlib
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from src.data.splits import validate_split,train_subset
from src.data.vision import ArrayImages
from src.ssl.views import Views,SupervisedImages
from src.ssl.augmentations import augmentation_hash
from src.utils.io import digest,file_hash
from src.utils.reproducibility import seed_worker

@lru_cache(maxsize=2)
def bundle(root,split_path):
    split=json.loads(Path(split_path).read_text(encoding='utf-8'));validate_split(split)
    # .targets is deliberately never accessed in this unlabelled data path.
    images=CIFAR10(root,train=True,download=False).data
    ids=split['roles']['train'];sums=np.zeros(3);squares=np.zeros(3);pixels=0
    for lo in range(0,len(ids),512):
        x=images[ids[lo:lo+512]].astype(np.float64)/255;sums+=x.sum((0,1,2));squares+=(x*x).sum((0,1,2));pixels+=np.prod(x.shape[:3])
    mean=sums/pixels;std=np.sqrt(squares/pixels-mean**2)
    sha=hashlib.sha256(images[ids].tobytes()).hexdigest()
    return images,split,mean.tolist(),std.tolist(),sha

def role_indices(split,role):
    if role=='cal':raise PermissionError('Lab4 cannot read cal; reserved untilLab8')
    return split['roles'][role]

def labelled_targets(root):return np.asarray(CIFAR10(root,train=True,download=False).targets)

def datasets(cfg):
    if cfg['data']['name']=='ssl_test_fixture':
        if cfg['experiment']['purpose']!='test':raise ValueError('Fixture forbidden in scientific runs')
        images=np.random.default_rng(11).integers(0,256,(8,32,32,3),dtype=np.uint8)
        ds={'train':Views(images,list(range(8)),[.5]*3,[.2]*3,cfg['augmentations'],cfg['experiment']['seed'])}
        return ds,{'data_fingerprint':digest({'fixture':11}),'split_fingerprint':digest({'fixture_train':8}),'synthetic_fixture':True}
    d=cfg['data'];images,split,mean,std,sha=bundle(d['root'],d['split_path']);seed=cfg['experiment']['seed']
    ids=role_indices(split,'train');ops=cfg['augmentations'];mode=cfg['model']['mode']
    if mode=='contrastive':
        train=Views(images,ids,mean,std,ops,seed);ds={'train':train}
    else:
        from src.analysis.label_efficiency import label_subsets
        labels=labelled_targets(d['root']);fraction=d['train_fraction'];ids=label_subsets(labels,ids,[fraction],cfg['experiment'].get('subset_seed',seed))[str(fraction)]
        train=SupervisedImages(images,ids,mean,std,ops,seed,labels=labels)
        ds={'train':train,'val':ArrayImages(images,labels,role_indices(split,'val'),mean,std)}
    info={'dataset':'official CIFAR10; immutable Lab0 split','data_fingerprint':digest({'train_pixels':sha,'indices':ids,'augmentation':ops,'mode':mode}),
          'split_fingerprint':split['fingerprint'],'actual_train_size':len(ids),'train_subset_indices':ids,
          'normalization':{'mean':mean,'std':std,'source':'43000train pixels only'},
          'augmentation_hash':augmentation_hash(ops),'split_sizes':{k:len(v) for k,v in split['roles'].items()},
          'pretraining_label_access':False if mode=='contrastive' else 'supervised baseline',
          'pretraining_image_role':'train' if mode=='contrastive' else None,'synthetic_fixture':False}
    if mode=='contrastive':
        n=cfg['train']['batch_size'];info.update(N=n,tau=cfg['contrastive']['tau'],negative_count=2*n-2,
            random_baseline_loss=float(np.log(2*n-1)),drop_last=True,discarded_per_epoch=len(ids)%n)
    info['worker_rng_protocol']='Checkpointed RandomSampler generator is independent of worker base-seed generator; augmentations are stateless(seed,epoch,image,view)'
    return ds,info

def loaders(cfg,ds,generator):
    ssl=cfg['model']['mode']=='contrastive';workers=cfg['data']['num_workers']
    kw={'num_workers':workers,'worker_init_fn':seed_worker,'persistent_workers':workers>0,'pin_memory':cfg['runtime']['device']=='cuda'}
    sampler=torch.utils.data.RandomSampler(ds['train'],generator=generator)
    worker_generator=torch.Generator().manual_seed(cfg['experiment']['seed']+4004)
    result={'train':torch.utils.data.DataLoader(ds['train'],batch_size=cfg['train']['batch_size'],sampler=sampler,
            generator=worker_generator,drop_last=ssl,**kw)}
    if 'val' in ds:result['val']=torch.utils.data.DataLoader(ds['val'],batch_size=256,shuffle=False,**{**kw,'persistent_workers':False})
    return result
