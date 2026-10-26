import importlib.metadata
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
import torch
from src.utils.io import digest, file_hash, write_json


def git_state(root=Path('.')):
    root = Path(root).resolve()
    def call(*args):
        # Trust only this explicit workspace invocation, without changing global Git trust settings.
        result = subprocess.run(['git','-c','safe.directory='+root.as_posix(),'-C',str(root),*args],capture_output=True,encoding='utf-8',errors='replace')
        return result.stdout.strip() if result.returncode == 0 else None
    commit = call('rev-parse','HEAD')
    # Run outputs cannot make the next identical experiment a different code condition.
    status = call('status','--porcelain','--','src','tests','scripts','configs','pyproject.toml')
    paths = []
    for directory in ['src','tests','scripts']:
        paths.extend(p for p in (Path(root)/directory).rglob('*.py') if '__pycache__' not in p.parts)
    pyproject = Path(root)/'pyproject.toml'
    if pyproject.exists():
        paths.append(pyproject)
    source_hash = digest({str(p.relative_to(root)).replace('\\','/'):file_hash(p) for p in sorted(paths)})
    return {'commit': commit, 'dirty': bool(status), 'source_hash': source_hash,
            'dirty_scope': 'src tests scripts configs pyproject.toml; excludes generated run/split metadata',
            'diff': call('diff','HEAD','--','src','tests','scripts','configs','pyproject.toml') or '',
            'source_snapshot':{str(p.relative_to(root)).replace('\\','/'):p.read_text(encoding='utf-8') for p in sorted(paths)} if (not commit or status) else None}


def environment(device):
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    return {'python': sys.version, 'pytorch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'os': platform.platform(),
            'cpu': platform.processor(), 'device': str(device),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
            'device_count': torch.cuda.device_count(), 'packages': packages,
            'cpu_temperature_c': None, 'cpu_temperature_status': 'sensor_not_available'}


def experiment_id(code, config, data_fingerprint, split_fingerprint):
    return digest({'code': {key:code[key] for key in ['commit','dirty','source_hash']},
                   'config_hash': digest(config), 'data': data_fingerprint,
                   'split': split_fingerprint, 'seed': config['experiment']['seed']})[:20]


def create_run(config, code, data_info, model, parent_run_id=None):
    lock = Path(config['runtime']['dependency_lock'])
    if not lock.is_file():
        raise FileNotFoundError('Generate the exact dependency lock before running')
    exp_id = experiment_id(code,config,data_info['data_fingerprint'],data_info['split_fingerprint'])
    run_id = exp_id+'-'+uuid.uuid4().hex
    path = Path(config['logging']['run_root'])/run_id
    path.mkdir(parents=True,exist_ok=False)
    device = next(model.parameters()).device
    env = environment(device)
    manifest = {'EXP_ID': exp_id, 'RUN_ID': run_id, 'started_at': datetime.now(timezone.utc).isoformat(),
                'profile': config['profile'], 'seed': config['experiment']['seed'],
                'git_commit': code['commit'], 'git_dirty': code['dirty'], 'source_hash': code['source_hash'],
                'git_dirty_scope': code['dirty_scope'], 'config_hash': digest(config), **data_info,
                'deterministic': config['experiment']['deterministic'], 'amp': config['train']['amp'],
                'num_workers': config['data']['num_workers'], 'dependency_lock_hash': file_hash(lock),
                'environment': env, 'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'resolved_config': 'config.resolved.yaml', 'parent_run_id': parent_run_id,
                'purpose': config['experiment']['purpose'], 'status': 'RUNNING'}
    write_json(path/'manifest.json',manifest)
    import yaml
    (path/'config.resolved.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    (path/'environment.txt').write_text('\n'.join(f'{k}: {v}' for k,v in env.items() if k != 'packages')+'\n'+'\n'.join(env['packages'])+'\n',encoding='utf-8')
    if code['dirty']:
        (path/'git.diff').write_text(code['diff']+'\nsource_hash: '+code['source_hash'],encoding='utf-8')
    if code.get('source_snapshot') is not None:
        write_json(path/'source_snapshot.json',code['source_snapshot'])
    return path,manifest
