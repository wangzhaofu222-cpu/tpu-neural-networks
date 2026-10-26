import hashlib
import json
from src.profile_gen import generate,lab_pick,LAB_OPTIONS


def test_L0_T09_profile_rules_and_unicode():
    for sid in ['unit-student-alpha','unit-student-beta','тест-студент']:
        p=generate(sid)
        assert p==generate(sid)
        h=hashlib.sha256(sid.encode('utf-8')).hexdigest()
        assert p['profile']['student_id_hash']==h
        assert p['profile']['id']=='P'+h[:6]
        assert p['profile']['base_seed']==int(h[:8],16)%2**31
        assert p['data']['train_fraction']==[.2,.25,.3,.35][int(h[8:10],16)%4]
        assert p['model']['width']==[48,64,96,128][int(h[10:12],16)%4]
        assert p['model']['depth']==[3,4,5][int(h[12:14],16)%3]
        assert p['train']['batch_size']==[64,128,256][int(h[14:16],16)%3]
        assert p['optimizer']['name']==['adamw','sgd'][int(h[16:18],16)%2]
        assert p['scheduler']['name']==['cosine','step'][int(h[18:20],16)%2]
        assert p['profile_batch_sizes']==[[1,8,32,64],[1,8,32,128],[1,16,64,256]][int(h[20:22],16)%3]
        for lab,fields in LAB_OPTIONS.items():
            hk=hashlib.sha256(f'{sid}:lab{lab}'.encode()).hexdigest()
            for i,(name,values) in enumerate(fields.items()):
                assert p[f'lab{lab}_{name}']==values[int(hk[2*i:2*i+2],16)%len(values)]
        assert sid not in json.dumps(p)
    assert generate('unit-student-alpha')!=generate('unit-student-beta')


def test_whitespace_only_is_removed():
    assert generate(' unit alpha ')==generate('unitalpha')
    assert generate('Alpha')!=generate('alpha')


def test_lab9_conflicting_fields_preserved():
    p=generate('unit-student-alpha')
    assert 'lab9_visual_encoder' in p and 'lab9_encoder' in p
    assert p['lab9_effective_visual_encoder']==p['lab9_visual_encoder']
    assert p['lab9_encoder_resolution']=='user_authorized_visual_encoder_precedence'
