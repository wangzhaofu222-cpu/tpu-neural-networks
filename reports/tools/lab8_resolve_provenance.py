"""Explicit byte-level data companions for immutable evaluation passports."""
from pathlib import Path
from reports.tools.lab8_common import R,read,registry
from src.utils.io import file_hash,write_json,digest

def main():
    raw={p.name:file_hash(p) for p in Path('data/cifar-10-batches-py').glob('*') if p.is_file()}
    data=read(R/'data_protocol.json');data['CIFAR10_raw_file_sha256']=raw;data['CIFAR10_raw_fingerprint']=digest(raw)
    data['passport_fingerprint_semantics']='Original manifest data_fingerprint is the frozen checkpoint input-source SHA256. Actual image-data bytes, split and role IDs are explicitly resolved in data_protocol.resolved.json; original RUN_ID/EXP_ID and passports remain unchanged.'
    write_json(R/'data_protocol.json',data)
    for key,path in registry().items():
        p=Path(path);m=read(p/'manifest.json');cfg=__import__('yaml').safe_load((p/'config.resolved.yaml').read_text(encoding='utf-8'))
        companion={'RUN_ID':p.name,'manifest_original_sha256':file_hash(p/'manifest.json'),'checkpoint_sha256':m['checkpoint_sha256'],'source_RUN_ID':m['source_RUN_ID'],
            'original_manifest_input_fingerprint':m['data_fingerprint'],'input_fingerprint_kind':'frozen_checkpoint_sha256','actual_CIFAR10_raw_file_sha256':raw,
            'actual_SVHN_test_sha256':data['SVHN_test']['sha256'],'split_sha256':data['split_sha256'],'split_fingerprint':data['split_fingerprint'],
            'evaluation':cfg['evaluation'],'normalization':m['normalization'],'course_protocol_status':'UNVERIFIED','cal_role_accessed':False}
        write_json(p/'data_protocol.resolved.json',companion)

if __name__=='__main__':main()
