"""Hash retained binary/run artifacts and export the actual final protocol."""
from pathlib import Path
import torch,yaml
from reports.tools.lab9_common import R,FEATURE,read
from src.utils.io import write_json,file_hash

def main():
    registry=read(R/'run_registry.json');checkpoints=[];runs=[];files={}
    for row in registry:
        root=Path(row['path']);cfg=yaml.safe_load((root/'config.resolved.yaml').read_text(encoding='utf-8'));passport=read(root/'manifest.json')
        runs.append({'kind':row['kind'],'RUN_ID':row['RUN_ID'],'seed':passport['seed'],'config_path':str(root/'config.resolved.yaml'),'model':cfg['model'],'evaluation_spec':cfg.get('evaluation_spec'),'purpose':passport['purpose'],'optimizer_steps':passport['global_step']})
        for p in sorted(root.rglob('*')):
            if p.is_file():files[str(p)]=file_hash(p)
        for p in sorted((root/'checkpoints').glob('*.pt')):
            s=torch.load(p,map_location='cpu',weights_only=False);checkpoints.append({'path':str(p),'sha256':files[str(p)],'RUN_ID':row['RUN_ID'],'seed':passport['seed'],'epoch':s['epoch'],'global_step':s['global_step'],'config_hash':s['config_hash'],'model_keys':len(s['model']),'optimizer_state_retained':'optimizer' in s,'RNG_retained':all(k in s for k in ['python_rng','numpy_rng','torch_rng','cuda_rng','loader_rng'])})
    assert file_hash(FEATURE)==read(R/'feature_cache.json')['sha256']
    write_json(R/'checkpoint_inventory.json',checkpoints)
    write_json(R/'artifact_inventory.json',{'scope':'All retained formal run files, including trusted binary checkpoints/predictions; raw archives have separate official MD5/SHA256 inventory','run_file_sha256':files,'immutable_pooled_input':{'path':str(FEATURE),'sha256':file_hash(FEATURE)},'checkpoint_count':len(checkpoints),'raw_data_policy':'Local data/binaries retained under existing ignore rules, readable identities and configurations versioned','official_archive_manifest':'acquisition/actor_*.json','source_encoder_manifest':'encoder_sources.json','all_raw_predictions_retained':all((Path(r['path'])/'predictions.pt').exists() for r in registry if r.get('evaluation'))})
    cfg={'captured_after_experiments':True,'authoritative_preregistration':read(R/'E1_preregistration.json'),'profile':read(R/'profile.json'),'data':{'manifest':'experiments/lab9/data_manifest.json','sha256':file_hash(R/'data_manifest.json'),'actor_split':'experiments/lab9/planned_actor_split.json','split_status':'TEACHER_SPLIT_UNVERIFIED','cal_media_read':False,'mel_normalization':'experiments/lab9/mel_normalization.json','clip_normalization':'experiments/lab9/feature_normalization.json'},'encoder_sources':read(R/'encoder_sources.json'),'feature_cache':read(R/'feature_cache.json'),'head_budget':{'epochs':200,'batch_size':64,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'selection':'val macro-F1 best checkpoint','early_stopping':False,'device':'CPU2threads'},'audio_budget':{'max_epochs':100,'patience':15,'selection':'val macro-F1','device':'CUDA RTX3060Laptop'},'main_selection':{'F':read(R/'E1_main.json')['F'],'U':read(R/'E1_main.json')['U'],'seeds':[1,2,3],'paired_direction':'F-U'},'late_registry_configs':'experiments/lab9/late_model_configs.json','runs':runs}
    (R/'config.final.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    qa=read(R/'figure_QA/images.json');qa.update(status='PASS',inspection='All12 figure families visually checked via three contact sheets; capacity/model/AUROC/SNR plots additionally inspected at native size. Overlapping capacity labels and long model labels corrected; no data edited.',source_hashes=read(R/'figures/artifact_hashes.json'));write_json(R/'figure_QA/images.json',qa)
    print('DELIVERY',len(checkpoints),'checkpoints',len(files),'run files',len(registry),'RUN_IDs')

if __name__=='__main__':main()
