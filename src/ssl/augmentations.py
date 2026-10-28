import torch
from torchvision import transforms as T
from src.utils.io import digest

def augmentation_hash(operations):return digest(operations)
def build_augmentation(operations,mean,std):
    items=[]
    for op in operations:
        name=op['name'];p=op['p']
        if name=='random_resized_crop':f=T.RandomResizedCrop(32,scale=tuple(op['scale']),antialias=True)
        elif name=='color_jitter':f=T.ColorJitter(**{k:op[k] for k in ['brightness','contrast','saturation','hue']})
        elif name=='flip':f=T.RandomHorizontalFlip(1.)
        elif name=='grayscale':f=T.RandomGrayscale(1.)
        elif name=='blur':f=T.GaussianBlur(op.get('kernel_size',3),sigma=tuple(op.get('sigma',[.1,2.])))
        else:raise ValueError(name)
        items.append(f if p==1 else T.RandomApply([f],p=p))
    return T.Compose(items+[T.ToTensor(),T.Normalize(mean,std)])

def apply_seeded(transform,image,seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return transform(image)
