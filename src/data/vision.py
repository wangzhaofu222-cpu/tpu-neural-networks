import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from src.data.splits import stratified_split, train_subset, validate_split
from src.utils.io import digest, file_hash, write_json
from src.utils.reproducibility import seed_worker


class ArrayImages(Dataset):
    def __init__(self, images, labels, indices, mean, std):
        self.images, self.labels, self.indices = images, labels, list(indices)
        self.mean = torch.tensor(mean,dtype=torch.float32)[:,None,None]
        self.std = torch.tensor(std,dtype=torch.float32)[:,None,None]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        x = torch.from_numpy(self.images[idx].copy()).permute(2,0,1).float()/255
        return (x-self.mean)/self.std, int(self.labels[idx])


def datasets(config):
    data = config['data']
    if data['name'] == 'test_fixture':
        if config['experiment']['purpose'] != 'test':
            raise ValueError('Synthetic fixtures are for tests only, never course results')
        g = torch.Generator().manual_seed(808)
        x = torch.randn(80,3,32,32,generator=g)
        y = torch.arange(80)%10
        ds = {'train':TensorDataset(x[:40],y[:40]), 'val':TensorDataset(x[40:60],y[40:60]), 'test':TensorDataset(x[60:],y[60:])}
        info = {'dataset':'test_fixture', 'data_fingerprint':digest({'x':x.tolist(),'y':y.tolist()}),
                'split_fingerprint':digest({'train':[0,40],'val':[40,60],'cal':[],'test':[60,80]}),
                'split_sizes':{'train':40,'val':20,'cal':0,'test':20}, 'synthetic_fixture':True}
        return ds,info
    if data['name'] != 'cifar10':
        raise ValueError('Only CIFAR-10 and explicit test fixtures are currently implemented')
    from torchvision.datasets import CIFAR10
    train = CIFAR10(data['root'],train=True,download=data['download'])
    test = CIFAR10(data['root'],train=False,download=data['download'])
    path = Path(data['split_path'])
    split = None
    if path.exists():
        split = json.loads(path.read_text(encoding='utf-8'))
        validate_split(split)
    counts = [data.get('val_per_class'),data.get('cal_per_class')]
    if any(value is None for value in counts):
        # Recover historical resolved configs from their saved shared split.
        if split is None:
            raise ValueError('A new CIFAR split requires explicit val_per_class and cal_per_class')
        classes = len(np.unique(train.targets))
        if any(len(split['roles'][role])%classes for role in ['val','cal']):
            raise ValueError('Legacy split does not have integral per-class role counts')
        counts = [len(split['roles'][role])//classes for role in ['val','cal']]
    expected = stratified_split(train.targets,data['split_seed'],*counts,len(test))
    if split is not None:
        if split != expected:
            raise ValueError('Shared split differs; refusing to overwrite')
    else:
        write_json(path,expected)
        split = expected
    indices = train_subset(train.targets,split['roles']['train'],data['train_fraction'],config['profile']['base_seed'])
    # Statistics use only the training role (never val/cal/test).
    sums,squares,pixels = np.zeros(3),np.zeros(3),0
    all_train = split['roles']['train']
    for start in range(0,len(all_train),512):
        chunk = train.data[all_train[start:start+512]].astype(np.float64)/255
        sums += chunk.sum(axis=(0,1,2))
        squares += (chunk*chunk).sum(axis=(0,1,2))
        pixels += chunk.shape[0]*chunk.shape[1]*chunk.shape[2]
    mean = sums/pixels
    std = np.sqrt(squares/pixels-mean**2)
    ds = {'train':ArrayImages(train.data,train.targets,indices,mean,std),
          'val':ArrayImages(train.data,train.targets,split['roles']['val'],mean,std),
          'test':ArrayImages(test.data,test.targets,split['roles']['test'],mean,std)}
    raw_files = sorted((Path(data['root'])/'cifar-10-batches-py').glob('*'))
    fingerprint = digest({p.name:file_hash(p) for p in raw_files if p.is_file()})
    info = {'dataset':'CIFAR-10 python official', 'version':'cifar-10-python',
            'data_fingerprint':fingerprint, 'split_fingerprint':split['fingerprint'],
            'subset_fingerprint':digest(indices), 'split_path':str(path),
            'split_sizes':{k:len(v) for k,v in split['roles'].items()},
            'actual_train_size':len(indices), 'normalization':{'mean':mean.tolist(),'std':std.tolist(),'source':'full train role only'},
            'synthetic_fixture':False}
    # The subset is part of experimental data conditions, not just a descriptive field.
    info['data_fingerprint'] = digest({'raw':fingerprint,'subset':info['subset_fingerprint']})
    if config['experiment']['lab']==2:
        from src.data.augmentations import AugmentedImages
        info['train_subset_indices']=indices
        info['augmentation']=data['augmentation']
        ds['train']=AugmentedImages(ds['train'],data['augmentation'],config['experiment']['seed'])
    return ds,info


def loaders(config, ds, generator):
    kw = {'batch_size':config['train']['batch_size'], 'num_workers':config['data']['num_workers'],
          'worker_init_fn':seed_worker, 'persistent_workers':False}
    # Independent validation generator prevents evaluation calls from perturbing shuffle RNG.
    val_g = torch.Generator().manual_seed(config['experiment']['seed']+1)
    return {'train':DataLoader(ds['train'],shuffle=True,generator=generator,**kw),
            'val':DataLoader(ds['val'],shuffle=False,generator=val_g,**kw)}
