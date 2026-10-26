import argparse
from src.config import load_config
from src.engine.runner import ExperimentRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--set',action='append',default=[])
    p.add_argument('--resume')
    p.add_argument('--stop-after-epoch',type=int)
    a = p.parse_args()
    print(ExperimentRunner(load_config(a.config,a.set)).run(a.resume,a.stop_after_epoch))


if __name__ == '__main__':
    main()
