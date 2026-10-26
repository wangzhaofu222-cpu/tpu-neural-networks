import copy
import numpy as np
import pytest
from src.data.splits import stratified_split,validate_split,train_subset,role_indices
from src.utils.io import file_hash


def test_L0_T02_all_roles_repeated_and_leak_free():
    labels=np.repeat(np.arange(10),5000)
    a=stratified_split(labels,1729,500,200,10000)
    assert a==stratified_split(labels,1729,500,200,10000)
    assert a!=stratified_split(labels,1730,500,200,10000)
    assert [len(a['roles'][r]) for r in ['train','val','cal','test']]==[43000,5000,2000,10000]
    assert len(set(a['roles']['train'])|set(a['roles']['val'])|set(a['roles']['cal']))==50000
    validate_split(a)


def test_cal_forbidden_and_leakage_rejected():
    a=stratified_split(np.repeat(np.arange(2),100),1,10,5,20)
    with pytest.raises(PermissionError):
        role_indices(a,'cal',4)
    assert len(role_indices(a,'cal',8,conformal=True))==10
    bad=copy.deepcopy(a)
    bad['roles']['val'].append(bad['roles']['train'][0])
    with pytest.raises(ValueError):
        validate_split(bad)


def test_nested_balanced_subsets():
    labels=np.repeat(np.arange(10),5000)
    split=stratified_split(labels,1729,500,200,10000)
    a=train_subset(labels,split['roles']['train'],.05,99)
    b=train_subset(labels,split['roles']['train'],.25,99)
    assert len(a)==2150
    assert set(a)<set(b)
    assert np.bincount(labels[a]).tolist()==[215]*10


def test_fingerprint_detects_changed_byte(tmp_path):
    p=tmp_path/'data.bin'
    p.write_bytes(b'abcdefgh')
    original=file_hash(p)
    p.write_bytes(b'abcdEfgh')
    assert file_hash(p)!=original


def test_cifar_consumer_uses_configured_roles_and_replays_legacy_split(fixture_config,tmp_path,monkeypatch):
    import torchvision.datasets
    from src.data.vision import datasets
    class LocalCIFAR:
        def __init__(self,root,train,download):
            size=1000 if train else 20
            self.data=np.random.default_rng(41+int(train)).integers(0,256,(size,32,32,3),dtype=np.uint8)
            self.targets=(np.arange(size)%10).tolist()
        def __len__(self): return len(self.targets)
    monkeypatch.setattr(torchvision.datasets,'CIFAR10',LocalCIFAR)
    cfg=copy.deepcopy(fixture_config)
    cfg['data'].update(name='cifar10',root=str(tmp_path/'data'),split_path=str(tmp_path/'split.json'),
                       train_fraction=.25,val_per_class=3,cal_per_class=2)
    _,info=datasets(cfg)
    assert info['split_sizes']=={'train':950,'val':30,'cal':20,'test':20}
    assert info['actual_train_size']==230
    legacy=copy.deepcopy(cfg)
    del legacy['data']['val_per_class']; del legacy['data']['cal_per_class']
    _,replayed=datasets(legacy)
    assert replayed==info
    cfg['data']['val_per_class']=4
    with pytest.raises(ValueError,match='Shared split differs'):
        datasets(cfg)
