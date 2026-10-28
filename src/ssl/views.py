import hashlib
import multiprocessing as mp
import torch
from PIL import Image
from src.ssl.augmentations import build_augmentation,apply_seeded

class Views(torch.utils.data.Dataset):
    """No label field or access. Per-view RNG is independent of split/model/workers."""
    def __init__(self,images,indices,mean,std,operations,seed):
        self.images,self.indices=images,list(indices);self.mean,self.std=mean,std
        self.operations,self.seed=operations,seed
        self.transform=build_augmentation(operations,mean,std)
        self.epoch=mp.get_context('spawn').Value('q',0)
    def __len__(self):return len(self.indices)
    def set_epoch(self,epoch):self.epoch.value=epoch
    def two_views(self,item,epoch=None):
        index=self.indices[item];image=Image.fromarray(self.images[index])
        epoch=self.epoch.value if epoch is None else epoch
        views=[]
        for view in [0,1]:
            seed=int.from_bytes(hashlib.sha256(f'lab4:augmentation:{self.seed}:{epoch}:{index}:{view}'.encode()).digest()[:8],'big')%(2**63)
            views.append(apply_seeded(self.transform,image,seed))
        return views[0],views[1],index
    def __getitem__(self,item):
        a,b,index=self.two_views(item)
        return torch.stack((a,b)),index

class SupervisedImages(Views):
    def __init__(self,*args,labels,**kwargs):super().__init__(*args,**kwargs);self.labels=labels
    def __getitem__(self,item):
        a,_,index=self.two_views(item)
        return a,int(self.labels[index])
