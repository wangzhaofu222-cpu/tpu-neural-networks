from .fgsm import fgsm, NormalizedModel
from .pgd import pgd


def attack(model,x,y,cfg):
    return (fgsm(model,x,y,cfg['epsilon']) if cfg['name']=='fgsm' else pgd(model,x,y,**{k:v for k,v in cfg.items() if k!='name'}))
