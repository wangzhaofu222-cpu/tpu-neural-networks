"""Run each canonical resource passport in a fresh process, without optimizer state."""
import argparse
from scripts.lab2_suite import base_config,execute,MODELS

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['cnn','vit']);a=p.parse_args()
    cfg=base_config();cfg['model']['name']=MODELS[a.arm];cfg['experiment']['role']='profiling'
    execute(f'Resource_{a.arm}_isolated',cfg,profile=True)

if __name__=='__main__':main()
