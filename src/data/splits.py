import numpy as np
from src.utils.io import digest


def stratified_split(labels, seed, val_per_class, cal_per_class, test_size):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    roles = {'train': [], 'val': [], 'cal': [], 'test': list(range(test_size))}
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label)).tolist()
        if len(indices) <= val_per_class+cal_per_class:
            raise ValueError('Insufficient examples for split')
        roles['val'].extend(indices[:val_per_class])
        roles['cal'].extend(indices[val_per_class:val_per_class+cal_per_class])
        roles['train'].extend(indices[val_per_class+cal_per_class:])
    for role in roles:
        roles[role].sort()
    result = {'schema': 1, 'seed': seed, 'method': 'per-class seeded permutation; sorted indices',
              'namespaces': {'train':'official_train','val':'official_train','cal':'official_train','test':'official_test'},
              'roles': roles}
    result['fingerprint'] = digest(result)
    validate_split(result)
    return result


def validate_split(split):
    roles = split['roles']
    for role,ids in roles.items():
        if len(ids) != len(set(ids)) or any(i<0 for i in ids):
            raise ValueError('Duplicate or negative split indices')
    for i,a in enumerate(roles):
        for b in list(roles)[i+1:]:
            if split['namespaces'][a] == split['namespaces'][b] and set(roles[a]) & set(roles[b]):
                raise ValueError(f'Data leakage: {a}/{b}')
    content = {k:v for k,v in split.items() if k != 'fingerprint'}
    if digest(content) != split['fingerprint']:
        raise ValueError('Split fingerprint mismatch')


def train_subset(labels, indices, fraction, seed):
    labels = np.asarray(labels)
    indices = np.asarray(indices)
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        group = indices[labels[indices] == label]
        count = int(len(group)*fraction)
        if count == 0:
            raise ValueError('Training fraction leaves an empty class')
        selected.extend(rng.permutation(group)[:count].tolist())
    return sorted(selected)


def role_indices(split, role, lab, conformal=False):
    if role == 'cal' and not (lab == 8 and conformal):
        raise PermissionError('cal is reserved for Lab 8 conformal prediction')
    return split['roles'][role]
