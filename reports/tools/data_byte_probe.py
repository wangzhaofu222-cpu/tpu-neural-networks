"""CF6: mutate one byte at the same sandbox-copy path, keeping official data untouched."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.models.small_cnn import build_model
from src.utils.io import file_hash,write_json
from src.utils.provenance import create_run,git_state
from src.utils.reproducibility import seed_all


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); cfg=load_config(a.run/'config.resolved.yaml')
    cfg['experiment'].update(name='lab0_CF06_same_path_byte_change',purpose='diagnostic',diagnostic_script_sha256=file_hash(__file__))
    candidate=ROOT/'tmp/CF06/data_batch_1'; candidate.parent.mkdir(parents=True,exist_ok=True)
    if candidate.exists(): raise RuntimeError('Probe path already exists; preserve it and choose a new dedicated path')
    original=ROOT/'data/cifar-10-batches-py/data_batch_1'; shutil.copy2(original,candidate)
    cfg['data']['probe_path']=str(candidate.relative_to(ROOT))
    seed_all(cfg['experiment']['seed'],True,2); model=build_model(cfg['model'])
    code=git_state(ROOT); split=json.loads((a.run/'manifest.json').read_text())['split_fingerprint']
    original_before=file_hash(original); before_hash=file_hash(candidate); size=candidate.stat().st_size
    before_path,before=create_run(cfg,code,{'dataset':'isolated byte probe, not a training dataset','data_fingerprint':before_hash,'split_fingerprint':split},model)
    with candidate.open('r+b') as stream:
        byte=stream.read(1); stream.seek(0); stream.write(bytes([byte[0]^1]))
    after_hash=file_hash(candidate)
    after_path,after=create_run(cfg,code,{'dataset':'isolated byte probe, not a training dataset','data_fingerprint':after_hash,'split_fingerprint':split},model)
    compared={key:before[key]==after[key] for key in ['EXP_ID','RUN_ID','config_hash','data_fingerprint','split_fingerprint','seed','git_commit','source_hash','profile']}
    result={'RUN_IDs':[before_path.name,after_path.name],'same_path_before_after':str(candidate),'same_filename':True,
            'size_unchanged':candidate.stat().st_size==size,'before_hash':before_hash,'after_hash':after_hash,
            'field_equality':compared,'official_data_untouched':original_before==file_hash(original),
            'status':'MEASURED_FINGERPRINT_INTERVENTION_NO_TRAINING'}
    for path,manifest in [(before_path,before),(after_path,after)]:
        manifest['status']='COMPLETED_FINGERPRINT_DIAGNOSTIC'; write_json(path/'manifest.json',manifest); write_json(path/'summary.json',result)
    write_json(a.out,result); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
