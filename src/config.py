import copy
import json
from pathlib import Path
import yaml


def merge(base, override):
    value = copy.deepcopy(base)
    for key,item in override.items():
        value[key] = merge(value[key],item) if isinstance(item,dict) and isinstance(value.get(key),dict) else copy.deepcopy(item)
    return value


def load_config(path, overrides=None):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if 'profile_path' in cfg:
        profile = yaml.safe_load(Path(cfg.pop('profile_path')).read_text(encoding='utf-8'))
        cfg = merge(cfg,profile)
        cfg['experiment']['seed'] = cfg['profile']['base_seed']
        # Resolve only before explicit, registered overrides (e.g. E2 seeds).
    for override in overrides or []:
        name,value = override.split('=',1)
        node = cfg
        parts = name.split('.')
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part],dict):
                raise ValueError(f'Unknown config path: {name}')
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f'Unknown config key: {name}')
        node[parts[-1]] = yaml.safe_load(value)
    validate(cfg)
    return cfg


def validate(cfg):
    for name in ['experiment','data','model','optimizer','scheduler','train','logging','profile','runtime']:
        if not isinstance(cfg.get(name),dict):
            raise ValueError(f'Missing section: {name}')
    if cfg['optimizer']['name'] not in ['adamw','sgd']:
        raise ValueError('Unsupported optimizer')
    if cfg['model']['name'] in {'deep_mlp','residual_mlp'}:
        model=cfg['model']
        if model['activation'] not in {'relu','tanh'} or model['normalization'] not in {'none','batchnorm','layernorm'}:
            raise ValueError('Unsupported MLP activation or normalization')
        if model['initialization'] not in {'he','xavier','gaussian'} or model['sigma'] <= 0:
            raise ValueError('Invalid MLP initialization')
        if any(type(model[k]) is not int or model[k] < 1 for k in ['depth','width']):
            raise ValueError('MLP depth and width must be positive integers')
        if model.get('normalization_position','before_activation') != 'before_activation':
            raise ValueError('Normalization must precede activation')
        alpha=model.get('alpha',1.)
        if model['name']=='residual_mlp' and alpha!='inverse_sqrt_depth' and (not isinstance(alpha,(int,float)) or alpha<=0):
            raise ValueError('Residual alpha must be positive or inverse_sqrt_depth')
    if cfg.get('diagnostics',{}).get('enabled'):
        d=cfg['diagnostics']
        if d['probe_batch_size']<64 or d['loss_subset_size']<d['probe_batch_size']:
            raise ValueError('Diagnostics require probe B>=64 and loss subset>=probe')
    if cfg['scheduler']['name'] not in ['cosine','step']:
        raise ValueError('Unsupported scheduler')
    if cfg['runtime']['device'] not in ['cpu','cuda']:
        raise ValueError('Device must be cpu or cuda')
    if cfg['train']['amp'] and cfg['runtime']['device'] != 'cuda':
        raise ValueError('AMP requires CUDA in this harness')
    if cfg['train']['epochs'] < 1 or cfg['train']['batch_size'] < 1:
        raise ValueError('epochs and batch_size must be positive')
    if not 0 < cfg['data']['train_fraction'] <= 1:
        raise ValueError('Invalid train fraction')
    if not 0 <= cfg['experiment']['seed'] < 2**32:
        raise ValueError('Invalid RNG seed')
    if cfg['data']['num_workers'] < 0 or cfg['runtime']['threads'] < 1:
        raise ValueError('Invalid resource configuration')
    if cfg['data']['name']=='cifar10':
        counts=[cfg['data'].get(k) for k in ['val_per_class','cal_per_class']]
        if any(x is not None for x in counts) and not all(type(x) is int and x>0 for x in counts):
            raise ValueError('Specify both positive integer per-class split counts')
        if cfg['experiment']['lab']==0 and cfg['experiment']['purpose'] in {'formal','budget'} and counts[0] is not None and counts!=[500,200]:
            raise ValueError('Lab 0 requires the teacher CIFAR role sizes')
    if cfg['logging']['save_every'] < 1:
        raise ValueError('save_every must be positive')
    if cfg['train'].get('max_steps') is not None:
        if type(cfg['train']['max_steps']) is not int or cfg['train']['max_steps']<1:
            raise ValueError('max_steps must be a positive integer')
    if cfg['experiment']['lab']==2:
        from src.data.augmentations import POLICIES
        if cfg['data'].get('augmentation') not in POLICIES:raise ValueError('Invalid Lab2 augmentation')
        if cfg['experiment']['purpose']=='formal' and cfg['experiment'].get('role') not in {'CF2','evaluation','profiling'}:
            if cfg['train']['max_steps']!=10000 or cfg['train']['batch_size']!=128:
                raise ValueError('Canonical Lab2 requires 10000 updates and batch128')
            if cfg['data']['augmentation']!='A1' or cfg['optimizer']['name']!='adamw':
                raise ValueError('Canonical Lab2 requires A1 and AdamW')
    if cfg['train']['accumulation_steps'] < 1:
        raise ValueError('accumulation_steps must be positive')
    if cfg['experiment']['lab'] == 9 and cfg.get('lab9_encoder_resolution') not in {'teacher_confirmed','user_authorized_visual_encoder_precedence'}:
        raise ValueError('Lab 9 encoder conflict requires teacher clarification')
    if not cfg['profile'].get('student_id_hash') or not cfg['profile'].get('id'):
        raise ValueError('A real generated profile is required')
    # Profile integrity can be checked from SHA256 directly for Lab 0 (without raw ID).
    h = cfg['profile']['student_id_hash']
    if len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
        raise ValueError('Invalid student hash')
    if cfg['profile']['id'] != 'P'+h[:6]:
        raise ValueError('Profile ID/hash mismatch')
    if cfg['profile']['base_seed'] != int(h[:8],16) % 2**31:
        raise ValueError('Profile base seed/hash mismatch')
    if cfg['experiment']['lab'] == 0 and cfg['experiment']['purpose'] in {'formal','budget'}:
        fields = [(cfg['model']['width'],[48,64,96,128],10,12),
                  (cfg['model']['depth'],[3,4,5],12,14),
                  (cfg['train']['batch_size'],[64,128,256],14,16),
                  (cfg['optimizer']['name'],['adamw','sgd'],16,18),
                  (cfg['scheduler']['name'],['cosine','step'],18,20),
                  (cfg['data']['train_fraction'],[.20,.25,.30,.35],8,10)]
        if any(actual != options[int(h[a:b],16)%len(options)] for actual,options,a,b in fields):
            raise ValueError('Lab 0 formal/budget configuration deviates from assigned profile')
    json.dumps(cfg,allow_nan=False)
