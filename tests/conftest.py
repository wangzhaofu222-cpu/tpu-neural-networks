import copy
import sys
import uuid
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config import load_config


def pytest_configure(config):
    if config.option.basetemp is None:
        # Windows sandbox temp ACLs may reject pytest's numbered directories.
        # Unique workspace directories preserve earlier test artifacts.
        (ROOT/'tmp').mkdir(exist_ok=True)
        config.option.basetemp = str(ROOT/'tmp'/('pytest-'+uuid.uuid4().hex))


@pytest.fixture
def fixture_config(tmp_path):
    cfg=load_config(ROOT/'configs/lab0.yaml')
    cfg=copy.deepcopy(cfg)
    cfg['experiment'].update(purpose='test',seed=123)
    cfg['data'].update(name='test_fixture',train_fraction=1.,num_workers=0)
    cfg['model'].update(width=8,depth=2)
    cfg['train'].update(batch_size=10,epochs=4)
    cfg['logging']['run_root']=str(tmp_path/'runs')
    cfg['runtime'].update(device='cpu',threads=1,max_seconds=30,dependency_lock=str(ROOT/'requirements.lock.txt'))
    return cfg
