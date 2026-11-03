import torch

def pack_clip_features(visual,audio,alignment,stats):
    """Restore canonical slots, then standardize with the fixed train clip statistics."""
    zv=visual['z'];za=audio['z'];dv=zv.shape[1];mean=torch.tensor(stats['mean']);std=torch.tensor(stats['std'])
    valid=alignment['visual_valid'].bool();indices=torch.tensor(visual['indices'],dtype=torch.long);v=torch.zeros(len(valid),dv)
    assert int(valid.sum())==len(zv) and indices.tolist()==alignment['visual_original_indices']
    v[indices]=(zv-mean[:dv])/std[:dv];a=(za-mean[dv:])/std[dv:]
    av=alignment['audio_valid'].bool();mask=(int(valid.any()),int(av.any()))
    if mask==(0,0):raise ValueError('Both modalities absent')
    return {'visual':v,'audio':a,'mask':mask,'align':{**alignment,'visual_valid':valid,'audio_valid':av}}


def load_feature_cache(path,*,encoder_sha256,dataset_hash,clip_id):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if obj['encoder_sha256']!=encoder_sha256 or obj['dataset_hash']!=dataset_hash or obj['clip_id']!=clip_id:
        raise ValueError('Feature cache identity/checkpoint mismatch')
    return obj
