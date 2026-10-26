"""Teacher's SHA256 rules. Raw student identifiers are never persisted."""
import argparse
import hashlib
import json
from pathlib import Path


def lab_pick(student_id: str, lab: int, a: int, b: int, options: list):
    hk = hashlib.sha256(f'{student_id}:lab{lab}'.encode('utf-8')).hexdigest()
    return options[int(hk[a:b], 16) % len(options)]


LAB_OPTIONS = {
    1: {'depths': [[4,16,32,64],[8,24,48,96]], 'bad_sigma': [.02,.05,.10], 'optimizer': ['sgd_momentum','adamw']},
    2: {'fractions': [[.05,.10,.25,.50,1.],[.05,.15,.30,.60,1.]],
        'corruptions': [['gaussian_noise','blur','patch_shuffle'],['contrast','occlusion','translation'],['gaussian_noise','occlusion','translation'],['blur','contrast','patch_shuffle']],
        'strong_aug': ['randaugment+cutmix','trivialaugment+mixup']},
    3: {'task': ['copy','retrieval','needle'], 'train_max': [64,128,256], 'depth': [2,4]},
    4: {'tau': [.1,.2,.5], 'third_aug': ['grayscale','flip','blur'], 'label_fractions': [[.01,.05,.25,1.],[.01,.10,.25,1.],[.02,.05,.25,1.]]},
    5: {'dataset': ['cora','citeseer','pubmed'], 'second_gnn': ['sage','gat'], 'depths': [[2,4,8,16],[2,3,6,12,24]]},
    6: {'target': ['eight_gaussians','four_corners','three_blobs'], 'eta': [[.01,.05,.1,.4],[.005,.02,.08,.5]], 'scale': [1.,2.,4.]},
    7: {'T_beta': [[500,1e-4,.04],[1000,1e-4,.02]], 'modes': [[6,.10],[8,.08],[10,.06]], 'rare_weight': [.01,.02,.04]},
    8: {'arch': ['cnn','vit'], 'corruption': ['gaussian_noise','gaussian_blur','contrast','occlusion'], 'pgd_eps_255': [2,4,8]},
    9: {'visual_encoder': ['cnn_lab2','vit_lab2','ssl_lab4'], 'p_drop': [.1,.3,.5], 'gate': ['vector','scalar']},
}


def generate(student_id):
    sid = ''.join(student_id.split())
    if not sid:
        raise ValueError('Student ID must not be empty')
    h = hashlib.sha256(sid.encode('utf-8')).hexdigest()
    def pick(a, b, values):
        return values[int(h[a:b], 16) % len(values)]
    p = {'profile': {'id': 'P'+h[:6], 'student_id_hash': h, 'base_seed': int(h[:8],16) % 2**31},
         'data': {'train_fraction': pick(8,10,[.20,.25,.30,.35]), 'num_workers': 0},
         'model': {'width': pick(10,12,[48,64,96,128]), 'depth': pick(12,14,[3,4,5])},
         'train': {'batch_size': pick(14,16,[64,128,256])},
         'optimizer': {'name': pick(16,18,['adamw','sgd'])},
         'scheduler': {'name': pick(18,20,['cosine','step'])},
         'profile_batch_sizes': pick(20,22,[[1,8,32,64],[1,8,32,128],[1,16,64,256]])}
    for lab, fields in LAB_OPTIONS.items():
        for i,(name, options) in enumerate(fields.items()):
            p[f'lab{lab}_{name}'] = lab_pick(sid,lab,2*i,2*i+2,options)
    p['lab9_encoder'] = lab_pick(sid,9,0,2,['resnet18','vit_small'])
    p['lab9_encoder_resolution'] = 'user_authorized_visual_encoder_precedence'
    p['lab9_effective_visual_encoder'] = p['lab9_visual_encoder']
    p['lab9_resolution_note'] = 'Use explicit upstream checkpoint source; preserve conflicting architecture field; not teacher-confirmed.'
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student-id', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    profile = generate(args.student_id)
    out = args.out or Path('configs/profiles')/(profile['profile']['id']+'.json')
    # Reject raw-ID filenames even if requested; PDF's filename example conflicts with its privacy rule.
    if ''.join(args.student_id.split()) in str(out):
        parser.error('Use profile ID rather than raw student ID in filenames')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {'.yaml','.yml'}:
        import yaml
        out.write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False),encoding='utf-8')
    else:
        out.write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'profile_id': profile['profile']['id'], 'path': str(out)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
