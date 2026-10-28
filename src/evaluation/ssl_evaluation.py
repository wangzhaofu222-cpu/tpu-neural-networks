"""Frozen-representation operations with the same provenance, models and runner."""
import copy
import json
from pathlib import Path
import torch
import yaml
from src.config import validate
from src.engine.checkpoint import load_checkpoint
from src.engine.runner import ExperimentRunner
from src.models.small_cnn import build_model
from src.evaluation.features import extract_role,extract_arrays,state_digest
from src.evaluation.knn import weighted_knn
from src.data.ssl_data import bundle,labelled_targets
from src.analysis.label_efficiency import label_subsets
from src.utils.io import digest,file_hash,write_json
from src.utils.provenance import create_run,git_state
from src.utils.reproducibility import seed_all

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def feature_cache(source,checkpoint='last.pt'):
    source=Path(source);cfg=yaml.safe_load((source/'config.resolved.yaml').read_text(encoding='utf-8'))
    seed_all(cfg['experiment']['seed'],True,cfg['runtime']['threads']);device=cfg['runtime']['device']
    model=build_model(cfg['model']).to(device);cp=source/'checkpoints'/checkpoint;state=load_checkpoint(cp,model)
    before=state_digest(model);config=copy.deepcopy(cfg);config['experiment']['role']='feature_extraction'
    config['feature_extraction']={'encoder_RUN_ID':source.name,'checkpoint':checkpoint,'checkpoint_sha256':file_hash(cp),'precision':'float32','random_augmentation':False}
    info={'data_fingerprint':digest(config['feature_extraction']),'split_fingerprint':state['split_fingerprint'],'synthetic_fixture':False}
    path,m=create_run(config,git_state(),info,model,source.name)
    cache={role:extract_role(model,cfg,role,device) for role in ['train','val','test']}
    images,split,mean,std,_=bundle(cfg['data']['root'],cfg['data']['split_path']);ids=split['roles']['train'][:256]
    again_h,again_z=extract_arrays(model,images,ids,mean,std,device,256)
    error_h=(cache['train']['h'][:256]-again_h).abs().max().item();error_z=(cache['train']['z_hat'][:256]-again_z).abs().max().item()
    assert error_h<=1e-6 and error_z<=1e-6 and state_digest(model)==before
    cache.update(encoder_RUN_ID=source.name,checkpoint=checkpoint,checkpoint_sha256=file_hash(cp),split_fingerprint=state['split_fingerprint'],
        fingerprint=digest({'checkpoint_sha256':file_hash(cp),'data_fingerprint':state['data_fingerprint'],'no_augmentation':True,'roles':['train','val','test']}))
    torch.save(cache,path/'features.pt')
    summary={'RUN_ID':path.name,'status':'COMPLETED_FEATURE_EXTRACTION','encoder_RUN_ID':source.name,'checkpoint':checkpoint,
        'checkpoint_sha256':file_hash(cp),'cache_path':(path/'features.pt').as_posix(),'feature_cache_sha256':file_hash(path/'features.pt'),
        'feature_fingerprint':cache['fingerprint'],'split_fingerprint':state['split_fingerprint'],'encoder_state_sha256':before,
        'repeated_extraction_max_error_h':error_h,'repeated_extraction_max_error_z':error_z,
        'role_sizes':{role:len(cache[role]['indices']) for role in ['train','val','test']},'cal_access':False,'optimizer_steps':0}
    write_json(path/'summary.json',summary);m.update(status=summary['status'],global_step=0);write_json(path/'manifest.json',m)
    return path,summary

def probe_config(base,cache_path,feature_kind,seed,fraction,indices,regularization,run_root):
    cfg=copy.deepcopy(base);cfg['experiment'].update(role='linear_probe',purpose=base['experiment']['purpose'] if base['experiment']['purpose'] in {'smoke','budget','test'} else 'formal',seed=seed)
    cfg['data']={'name':'ssl_features','root':base['data']['root'],'train_fraction':fraction,'num_workers':0,
        'feature_path':str(cache_path),'feature_kind':feature_kind,'label_indices':indices,'subset_seed':seed}
    cfg['model']={'name':'linear_probe','input_dim':512 if feature_kind=='h' else 128,'classes':10}
    cfg['optimizer'].update(name='sgd',lr=base['evaluation']['probe_lr'],weight_decay=regularization,momentum=.9)
    cfg['scheduler'].update(name='cosine',unit='epoch',warmup_epochs=0)
    cfg['train'].update(epochs=base['evaluation']['probe_epochs'],batch_size=65536,amp=False,validation_at_end_only=True)
    cfg['train'].pop('max_steps',None);cfg['representation']['enabled']=False
    cfg['logging'].update(run_root=run_root,save_every=cfg['train']['epochs'],keep_epochs=[])
    cfg['probe_protocol']={'pair_seed':seed,'subset_rng':'independent local NumPy PCG64(seed)','initialization_rng':'Torch seed','fraction':fraction,
        'feature_kind':feature_kind,'regularization':regularization,'regularization_selection':'max final validation accuracy, ties smallerL2',
        'full_batch':True,'standardization':'selected train mean/std only; std floor1e-5','encoder_frozen':'offlinefeatures; no encoder in optimizer'}
    validate(cfg);return cfg

def evaluate_probe(base,feature_run,seed=1,fraction=1.,feature_kind='h',label='probe',run_root='runs/lab4'):
    feature_run=Path(feature_run);meta=read(feature_run/'summary.json');cache=torch.load(feature_run/'features.pt',map_location='cpu',weights_only=False)
    before_sha=file_hash(Path('runs/lab4')/cache['encoder_RUN_ID']/'checkpoints'/cache['checkpoint'])
    labels=labelled_targets(base['data']['root']);indices=label_subsets(labels,cache['train']['indices'],[fraction],seed)[str(fraction)]
    candidates=[];device=base['runtime']['device']
    for regularization in base['evaluation']['regularization_grid']:
        cfg=probe_config(base,feature_run/'features.pt',feature_kind,seed,fraction,indices,regularization,run_root)
        p=ExperimentRunner(cfg).run();s=read(p/'summary.json')
        candidates.append({'regularization':regularization,'validation_accuracy':s['validation_metric'],'RUN_ID':p.name,'path':p.as_posix()})
    winner=max(candidates,key=lambda r:(r['validation_accuracy'],-r['regularization']));chosen=Path(winner['path'])
    cfg=yaml.safe_load((chosen/'config.resolved.yaml').read_text(encoding='utf-8'));manifest=read(chosen/'manifest.json')
    classifier=build_model(cfg['model']).to(device);load_checkpoint(chosen/'checkpoints/last.pt',classifier);classifier.eval()
    norm=manifest['normalization'];mean=torch.tensor(norm['feature_mean'],device=device);std=torch.tensor(norm['feature_std'],device=device)
    with torch.no_grad():
        x=cache['test'][feature_kind].to(device);target=cache['test']['labels'].to(device)
        predictions=classifier((x-mean)/std).argmax(1);accuracy=(predictions==target).sum().item()/len(target)
        # kNN always uses the complete train reference, independent of probe label fraction.
        knn_path=feature_run/f'knn_{feature_kind}_k{base["evaluation"]["knn_k"]}_tau{base["evaluation"]["knn_tau"]}.pt'
        if knn_path.exists():knn=torch.load(knn_path,map_location=device,weights_only=True)
        else:
            knn=weighted_knn(cache['train'][feature_kind].to(device),cache['train']['labels'].to(device),x,
                             base['evaluation']['knn_k'],base['evaluation']['knn_tau'])
            torch.save(knn.cpu(),knn_path)
        knn_accuracy=(knn==target).sum().item()/len(target)
    assert before_sha==file_hash(Path('runs/lab4')/cache['encoder_RUN_ID']/'checkpoints'/cache['checkpoint'])
    info={'data_fingerprint':cache['fingerprint'],'split_fingerprint':cache['split_fingerprint'],'synthetic_fixture':False,
        'encoder_RUN_ID':cache['encoder_RUN_ID'],'encoder_full_state_sha256':meta['encoder_state_sha256'],'label_indices':indices,
        'parent_checkpoint_sha256':before_sha,'chosen_probe_RUN_ID':winner['RUN_ID']}
    evalcfg=copy.deepcopy(cfg);evalcfg['experiment']['role']='probe_evaluation';evalcfg['evaluation_result_label']=label
    path,m=create_run(evalcfg,git_state(),info,classifier,cache['encoder_RUN_ID'])
    summary={'RUN_ID':path.name,'status':'COMPLETED_PROBE_EVALUATION','label':label,'encoder_RUN_ID':cache['encoder_RUN_ID'],
        'feature_RUN_ID':feature_run.name,'feature_kind':feature_kind,'checkpoint':cache['checkpoint'],'pair_seed':seed,'fraction':fraction,
        'test_accuracy':accuracy,'knn_test_accuracy':knn_accuracy,'validation_accuracy':winner['validation_accuracy'],
        'selected_regularization':winner['regularization'],'candidates':candidates,'label_indices':indices,
        'label_subset_fingerprint':digest(indices),'encoder_full_state_unchanged':True,'encoder_checkpoint_sha256':before_sha,
        'encoder_full_state_sha256':meta['encoder_state_sha256'],'optimizer_steps':0,'test_images':len(target)}
    torch.save({'test_predictions':predictions.cpu(),'test_targets':target.cpu(),'knn_predictions':knn.cpu()},path/'predictions.pt')
    write_json(path/'summary.json',summary);m.update(status=summary['status'],global_step=0);write_json(path/'manifest.json',m)
    return path,summary
