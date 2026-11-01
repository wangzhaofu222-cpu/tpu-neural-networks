"""Structured evaluation profile; undefined metrics are explicit."""
from src.utils.io import write_json


def write_profile(path, *, clean,corruption,adversarial,selective,ood,protocol):
    result=dict(clean=clean,corruption=corruption,adversarial=adversarial,selective=selective,ood=ood,protocol=protocol,
                undefined=[{'metric':m,'data':'SVHN test','reason':'no true class in CIFAR-10 label space'} for m in ['accuracy','ece','nll','brier']])
    write_json(path,result);return result
