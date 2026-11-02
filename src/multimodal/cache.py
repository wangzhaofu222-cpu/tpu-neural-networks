import torch


def load_feature_cache(path,*,encoder_sha256,dataset_hash,clip_id):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if obj['encoder_sha256']!=encoder_sha256 or obj['dataset_hash']!=dataset_hash or obj['clip_id']!=clip_id:
        raise ValueError('Feature cache identity/checkpoint mismatch')
    return obj
