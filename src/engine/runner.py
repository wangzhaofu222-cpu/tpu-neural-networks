import json
import math
import shutil
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from src.metrics.objective import correct_count,objective
from src.data.vision import datasets,loaders
from src.models.small_cnn import build_model
from src.engine.checkpoint import load_checkpoint,save_checkpoint
from src.metrics.classification import evaluate_loader
from src.utils.io import digest,write_json
from src.utils.provenance import create_run,git_state
from src.utils.reproducibility import seed_all


class ExperimentRunner:
    def __init__(self,config):
        self.config = config

    def run_sequence_profile(self,length,dtype=torch.float32):
        from src.engine.sequence_evaluation import profile
        return profile(self.config,length,dtype)

    @staticmethod
    def evaluate_sequence(source,length,rule='P2',role='test',diagnostics=False):
        from src.engine.sequence_evaluation import evaluate
        return evaluate(source,length,rule,role,diagnostics)

    @staticmethod
    def evaluate_checkpoint(source,corruption='clean',level=0):
        from src.engine.vision_evaluation import evaluate_checkpoint
        return evaluate_checkpoint(source,corruption,level)

    def run(self,resume=None,stop_after_epoch=None,weights_only_epoch=None,initialize_only=False):
        cfg = self.config
        ssl_mode=cfg['model']['name']=='simclr' and cfg['model']['mode']=='contrastive'
        device = torch.device(cfg['runtime']['device'])
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
        code = git_state()
        if cfg['experiment']['purpose'] == 'formal' and (not code['commit'] or code['dirty']):
            raise RuntimeError('Formal experiments require committed clean source/config')
        seed_all(cfg['experiment']['seed'],cfg['experiment']['deterministic'],cfg['runtime']['threads'])
        ds,data_info = datasets(cfg)
        g = torch.Generator().manual_seed(cfg['experiment']['seed'])
        dl = loaders(cfg,ds,g)
        model = build_model(cfg['model']).to(device)
        opt_cfg = cfg['optimizer']
        options = {'lr':opt_cfg['lr'],'weight_decay':opt_cfg['weight_decay']}
        optimizer = (torch.optim.AdamW(model.parameters(),**options) if opt_cfg['name']=='adamw' else
                     torch.optim.Adam(model.parameters(),**options) if opt_cfg['name']=='adam' else
                     torch.optim.SGD(model.parameters(),momentum=opt_cfg['momentum'],**options))
        total_epochs = cfg['train']['epochs']
        step_limit = cfg['train'].get('max_steps')
        step_schedule = cfg['scheduler'].get('unit') == 'optimizer_step'
        if step_limit:
            total_epochs = math.ceil(step_limit*cfg['train']['accumulation_steps']/len(dl['train']))+1
        warmup = cfg['scheduler']['warmup_epochs']
        def multiplier(epoch):
            if cfg['scheduler']['name']=='constant':return 1.
            if step_schedule:
                warm_steps=math.ceil(step_limit*cfg['scheduler']['warmup_fraction'])
                if epoch<warm_steps:return (epoch+1)/max(1,warm_steps)
                return .5*(1+math.cos(math.pi*min(1,(epoch-warm_steps)/max(1,step_limit-warm_steps))))
            if warmup and epoch < warmup:
                return (epoch+1)/warmup
            if cfg['scheduler']['name'] == 'step':
                return cfg['scheduler']['gamma'] ** ((epoch-warmup)//cfg['scheduler']['step_size'])
            progress = min(1,max(0,(epoch-warmup)/max(1,total_epochs-warmup)))
            return .5*(1+math.cos(math.pi*progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
        scaler = torch.amp.GradScaler('cuda',enabled=cfg['train']['amp'])
        start_epoch,global_step,best,best_epoch = 0,0,-1.,None
        parent = None
        state = None
        if resume:
            if weights_only_epoch is not None:
                if cfg['experiment'].get('resume_ablation') != 'weights_only':
                    raise ValueError('Weights-only ablation must be registered in resolved config')
                if not 0 < weights_only_epoch < total_epochs:
                    raise ValueError('Invalid ablation continuation epoch')
                truncated=torch.load(resume,map_location='cpu',weights_only=False)
                if set(truncated) != {'model'}:
                    raise ValueError('Controlled ablation must load a genuinely model-only checkpoint')
                model.load_state_dict(truncated['model'])
                start_epoch=weights_only_epoch
                global_step=weights_only_epoch*math.ceil(len(dl['train'])/cfg['train']['accumulation_steps'])
                parent=Path(resume).parent.parent.name
            else:
                state = load_checkpoint(resume,model,optimizer,scheduler,scaler,g)
        if state is not None:
            if state.get('partial_epoch'):
                raise ValueError('Final partial data-epoch checkpoint is for inference; resume requires a complete data-epoch boundary')
            config_match=state['config_hash']==digest(cfg)
            if not config_match and cfg['experiment']['lab']==3:
                import yaml,copy
                prior=yaml.safe_load((Path(resume).parent.parent/'config.resolved.yaml').read_text(encoding='utf-8'))
                expected=copy.deepcopy(prior);expected['train']['max_steps']=cfg['train']['max_steps']
                config_match=(expected==cfg and cfg['train']['max_steps']>prior['train']['max_steps'] and cfg['scheduler']['name']=='constant')
            if not config_match or state['data_fingerprint'] != data_info['data_fingerprint'] or state['split_fingerprint'] != data_info['split_fingerprint']:
                raise ValueError('Resume config/data/split mismatch')
            if state['source_hash'] != code['source_hash']:
                raise ValueError('Resume source changed')
            start_epoch,global_step = state['epoch'],state['global_step']
            best,best_epoch,parent = state['best_accuracy'],state['best_epoch'],state['run_id']
        path,manifest = create_run(cfg,code,data_info,model,parent)
        (path/'checkpoints').mkdir()
        if resume:
            parent_path = Path(resume).parent.parent
            for name in ['metrics.jsonl','steps.jsonl']+(['gradients.jsonl'] if cfg['experiment']['lab']==5 else []):
                if (parent_path/name).exists():
                    # Keep only entries that logically precede this checkpoint.
                    records = [json.loads(line) for line in (parent_path/name).read_text().splitlines()]
                    records = [r for r in records if r['epoch'] <= start_epoch]
                    (path/name).write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
            if (parent_path/'checkpoints/best.pt').exists():
                shutil.copy2(parent_path/'checkpoints/best.pt',path/'checkpoints/best.pt')
        first_checksum = state.get('first_batch_checksum') if state else None
        wall_start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        budget = cfg['runtime']['max_seconds']
        diagnostics = None
        if cfg.get('diagnostics', {}).get('enabled'):
            from src.diagnostics.trainability import TrainingDiagnostics
            diagnostics = TrainingDiagnostics(model, ds['train'], cfg, path)
        representation_monitor=None
        if cfg.get('representation',{}).get('enabled'):
            from src.diagnostics.ssl_monitor import RepresentationMonitor
            representation_monitor=RepresentationMonitor(model,ds['train'],cfg,path)
        if cfg['model']['name']=='simclr' and not resume:
            extra0={'best_accuracy':best,'best_epoch':best_epoch,'run_id':path.name,'config_hash':digest(cfg),
                    'data_fingerprint':data_info['data_fingerprint'],'split_fingerprint':data_info['split_fingerprint'],
                    'source_hash':code['source_hash'],'first_batch_checksum':None}
            save_checkpoint(path/'checkpoints/epoch_000.pt',model,optimizer,scheduler,scaler,0,0,g,**extra0)
            if representation_monitor:representation_monitor.observe(0)
            if initialize_only:
                shutil.copy2(path/'checkpoints/epoch_000.pt',path/'checkpoints/last.pt')
                manifest.update(status='COMPLETED_INITIALIZATION',global_step=0,completed_epoch=0)
                write_json(path/'manifest.json',manifest)
                write_json(path/'summary.json',{'RUN_ID':path.name,'status':'COMPLETED_INITIALIZATION','global_step':0,'optimizer_steps':0,
                    'representation':representation_monitor.last['representation'] if representation_monitor else None})
                return path
        completed_epoch = start_epoch
        samples_processed=state.get('samples_processed',0) if state else 0
        checkpoint_step=global_step
        console_step=global_step
        status = 'RUNNING'
        try:
            for epoch in range(start_epoch,total_epochs):
                epoch_start = time.perf_counter()
                model.train()
                if hasattr(ds['train'],'set_epoch'):ds['train'].set_epoch(epoch)
                loss_sum,correct,count = 0.,0,0
                similarity_sums={'positive_similarity':0.,'negative_similarity':0.}
                lr_used = optimizer.param_groups[0]['lr']
                epoch_initial_step = global_step
                optimizer.zero_grad(set_to_none=True)
                accumulation = cfg['train']['accumulation_steps']
                for batch_index,(x,y) in enumerate(dl['train']):
                    optimizer_step_skipped = False
                    if budget and time.perf_counter()-wall_start > budget:
                        raise TimeoutError('Configured wall time exceeded; no incomplete-epoch checkpoint claimed')
                    if first_checksum is None:
                        if cfg['experiment']['lab'] in {4,5}:
                            import hashlib
                            first_checksum=hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()+y.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        else:first_checksum = digest({'images':x.tolist(),'labels':y.tolist()})
                    x,y = x.to(device),y.to(device)
                    if step_limit:lr_used=optimizer.param_groups[0]['lr']
                    if cfg['experiment']['lab']==2:
                        from src.data.augmentations import mix_batch
                        x,y,mixing=mix_batch(x,y,cfg['data']['augmentation'],cfg['model']['classes'])
                    if diagnostics:
                        diagnostics.begin(batch_index, len(dl['train']))
                    with torch.autocast(device_type=device.type,enabled=cfg['train']['amp']):
                        logits = model(x)
                        if ssl_mode:
                            from src.ssl.losses import nt_xent
                            loss,contrast_stats=nt_xent(*logits,cfg['contrastive']['tau'])
                        else:loss = objective(logits,y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    # Normalize the final, possibly shorter accumulation group correctly.
                    group_start = (batch_index//accumulation)*accumulation
                    group_length = min(accumulation,len(dl['train'])-group_start)
                    group_examples = sum(min(cfg['train']['batch_size'],len(ds['train'])-j*cfg['train']['batch_size'])
                                         for j in range(group_start,group_start+group_length))
                    scaled_loss = loss if accumulation == 1 else loss*len(y)/group_examples
                    scaler.scale(scaled_loss).backward()
                    if cfg['experiment']['lab']==5 and cfg['model']['name']=='graph_network':
                        from src.diagnostics.oversmoothing import gradient_record
                        self._append(path/'gradients.jsonl',{'epoch':epoch+1,**gradient_record(model)})
                    if diagnostics:
                        diagnostics.end(epoch+1, batch_index)
                        if not bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()):
                            raise FloatingPointError('Non-finite parameter gradient')
                    if (batch_index+1)%accumulation==0 or batch_index+1==len(dl['train']):
                        previous_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer_step_skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
                        optimizer.zero_grad(set_to_none=True)
                        if not optimizer_step_skipped:
                            global_step += 1
                            if step_schedule:scheduler.step()
                    hits,targets=(0,len(y)) if ssl_mode else correct_count(logits,y)
                    if ssl_mode:
                        for key in similarity_sums:similarity_sums[key]+=contrast_stats[key]*len(y)
                    loss_sum += loss.item()*targets
                    correct += hits
                    count += targets
                    samples_processed += len(y)
                    self._append(path/'steps.jsonl',{'epoch':epoch+1,'batch':batch_index,'global_step':global_step,'train_loss':loss.item(),'learning_rate':lr_used,'optimizer_step_skipped':optimizer_step_skipped,
                        **(contrast_stats if ssl_mode else {})})
                    if step_limit and global_step>=step_limit:break
                finished_steps=bool(step_limit and global_step>=step_limit)
                partial_epoch=batch_index+1<len(dl['train'])
                evaluate_at_end=finished_steps or (not step_limit and epoch+1==total_epochs)
                val = (evaluate_loader(model,dl['val'],device) if not ssl_mode and (not cfg['train'].get('validation_at_end_only') or evaluate_at_end) else {'loss':None,'accuracy':None})
                if diagnostics and not math.isfinite(val['loss']):
                    raise FloatingPointError('Non-finite validation loss')
                if global_step > epoch_initial_step and not step_schedule:
                    scheduler.step()
                completed_epoch = epoch if partial_epoch else epoch+1
                record = {'epoch':completed_epoch,'global_step':global_step,'train_loss':loss_sum/count,
                          'validation_loss':val['loss'],'train_metric':correct/count,'validation_metric':val['accuracy'],
                          'learning_rate':lr_used,'epoch_time':time.perf_counter()-epoch_start,
                          'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
                if ssl_mode:
                    record.update({k:v/count for k,v in similarity_sums.items()})
                    record['train_metric']=None
                    record['train_metric_name']='not_applicable_SSL_uses_loss_and_similarities'
                if representation_monitor and (completed_epoch%cfg['representation']['interval']==0 or completed_epoch==total_epochs):
                    record['representation']=representation_monitor.observe(completed_epoch)['representation']
                    record['epoch_time']=time.perf_counter()-epoch_start
                if step_limit:
                    record.update(data_epochs=samples_processed/len(ds['train']),samples_processed=samples_processed,
                                  partial_epoch=partial_epoch,target_optimizer_steps=step_limit)
                self._append(path/'metrics.jsonl',record)
                improved = val['accuracy'] is not None and val['accuracy'] > best
                if improved:
                    best,best_epoch = val['accuracy'],completed_epoch
                extra = {'best_accuracy':best,'best_epoch':best_epoch,'run_id':manifest['RUN_ID'],
                         'config_hash':digest(cfg),'data_fingerprint':data_info['data_fingerprint'],
                         'split_fingerprint':data_info['split_fingerprint'],'source_hash':code['source_hash'],
                         'first_batch_checksum':first_checksum}
                if step_limit:extra.update(partial_epoch=partial_epoch,samples_processed=samples_processed)
                save_due=(global_step-checkpoint_step>=cfg['logging'].get('save_every_steps',1000)) if step_limit else completed_epoch%cfg['logging']['save_every']==0
                if save_due or finished_steps or completed_epoch==total_epochs or completed_epoch==stop_after_epoch:
                    save_checkpoint(path/'checkpoints/last.pt',model,optimizer,scheduler,scaler,completed_epoch,global_step,g,**extra)
                    checkpoint_step=global_step
                if improved:
                    save_checkpoint(path/'checkpoints/best.pt',model,optimizer,scheduler,scaler,completed_epoch,global_step,g,**extra)
                if completed_epoch in cfg['logging'].get('keep_epochs',[]):
                    save_checkpoint(path/f'checkpoints/epoch_{completed_epoch:03}.pt',model,optimizer,scheduler,scaler,completed_epoch,global_step,g,**extra)
                if cfg['experiment']['lab']==4 and cfg['model']['name']=='simclr' and cfg['runtime'].get('gpu_stop_temperature') and device.type=='cuda':
                    import subprocess
                    measured=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
                    self._append(path/'hardware.jsonl',{'epoch':completed_epoch,'nvidia_smi':measured.stdout.strip(),'returncode':measured.returncode})
                    if measured.returncode==0 and float(measured.stdout.split(',')[0])>=cfg['runtime']['gpu_stop_temperature']:
                        raise RuntimeError('GPU temperature safety threshold reached; complete epoch checkpoint preserved')
                write_json(path/'summary.json',{**record,'RUN_ID':manifest['RUN_ID'],'EXP_ID':manifest['EXP_ID'],
                           'best_accuracy':best,'best_epoch':best_epoch,'first_batch_checksum':first_checksum,
                           'checkpoint_for_final_metric':'checkpoints/last.pt','purpose':cfg['experiment']['purpose']})
                console_due=(not step_limit or finished_steps or global_step-console_step>=cfg['logging'].get('console_every_steps',500))
                if cfg['experiment']['lab']==5:console_due=(epoch+1==total_epochs or (epoch+1)%50==0)
                if cfg['model']['name']=='linear_probe':console_due=(epoch+1==total_epochs)
                if console_due:
                    console_record={**record}
                    if cfg['experiment']['lab']==4 and 'representation' in console_record:
                        console_record['representation']={name:{k:v for k,v in stats.items() if k!='eigenvalues'} for name,stats in console_record['representation'].items()}
                    print(json.dumps({'RUN_ID':manifest['RUN_ID'],**console_record}),flush=True)
                    console_step=global_step
                if finished_steps:
                    status='COMPLETED'
                    break
                if stop_after_epoch and completed_epoch >= stop_after_epoch and completed_epoch<total_epochs:
                    status = 'INTERRUPTED_AT_EPOCH_BOUNDARY'
                    break
                pause_file=cfg['runtime'].get('pause_file')
                if pause_file and Path(pause_file).exists() and completed_epoch<total_epochs:
                    # User-requested pause is checked only after a complete recoverable checkpoint.
                    save_checkpoint(path/'checkpoints/last.pt',model,optimizer,scheduler,scaler,completed_epoch,global_step,g,**extra)
                    status='INTERRUPTED_AT_EPOCH_BOUNDARY'
                    break
            else:
                status = 'COMPLETED'
        except Exception as exc:
            status = 'FAILED'
            exc.run_path=str(path)
            write_json(path/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed_epoch':completed_epoch})
            if not (diagnostics and isinstance(exc, FloatingPointError)):
                raise
        finally:
            training_seconds=time.perf_counter()-wall_start
            if diagnostics:
                existing = json.loads((path/'summary.json').read_text(encoding='utf-8')) if (path/'summary.json').exists() else {
                    'RUN_ID': manifest['RUN_ID'], 'EXP_ID': manifest['EXP_ID'], 'purpose': cfg['experiment']['purpose']}
                existing['trainability'] = diagnostics.finish(failed=status == 'FAILED')
                write_json(path/'summary.json', existing)
            if cfg['experiment']['lab']==2 and status=='COMPLETED' and cfg['logging'].get('profile_on_finish'):
                from src.profiling.model_profiler import resource_profile
                resources=resource_profile(model,cfg['model'],device)
                resources['training_seconds']=training_seconds
                resources['training_peak_gpu_memory_bytes']=record['peak_gpu_memory_bytes']
                write_json(path/'resources.json',resources)
            manifest.update(status=status,completed_epoch=completed_epoch,global_step=global_step,
                            first_batch_checksum=first_checksum,wall_seconds=time.perf_counter()-wall_start)
            write_json(path/'manifest.json',manifest)
            if (path/'summary.json').exists():
                summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
                summary['status']=status
                if step_limit or cfg['experiment']['lab'] in {4,5}:summary['training_seconds']=training_seconds
                write_json(path/'summary.json',summary)
        return path

    def run_profile(self):
        from src.profiling.model_profiler import resource_profile
        cfg=self.config;code=git_state()
        if cfg['experiment']['purpose']=='formal' and code['dirty']:raise RuntimeError('Uncommitted source')
        seed_all(cfg['experiment']['seed'],True,cfg['runtime']['threads'])
        device=torch.device(cfg['runtime']['device']);model=build_model(cfg['model']).to(device)
        info={'dataset':'explicit resource-profiling tensors; no training',
              'data_fingerprint':digest({'shape':[3,32,32],'distribution':'normal','seed':cfg['experiment']['seed']}),
              'split_fingerprint':digest({'no_dataset_split_for_model_profiling':True}),'synthetic_fixture':False}
        path,manifest=create_run(cfg,code,info,model)
        result=resource_profile(model,cfg['model'],device)
        result.update(RUN_ID=path.name,status='COMPLETED_RESOURCE_PROFILE',optimizer_steps=0)
        write_json(path/'resources.json',result);write_json(path/'summary.json',result)
        self._append(path/'metrics.jsonl',{'operation':'resource_profile','MACs':result['MACs'],'parameters':result['parameters']})
        manifest.update(status='COMPLETED_RESOURCE_PROFILE',global_step=0,completed_epoch=0)
        write_json(path/'manifest.json',manifest)
        return path

    def run_probe(self):
        """No optimizer: same data, model factory, seeding and provenance as training."""
        from src.diagnostics.trainability import TrainingDiagnostics
        cfg = self.config
        code = git_state()
        if cfg['experiment']['purpose'] == 'formal' and (not code['commit'] or code['dirty']):
            raise RuntimeError('Formal probes require committed clean source/config')
        seed_all(cfg['experiment']['seed'], cfg['experiment']['deterministic'], cfg['runtime']['threads'])
        start = time.perf_counter()
        ds, info = datasets(cfg)
        model = build_model(cfg['model']).to(cfg['runtime']['device'])
        path, manifest = create_run(cfg, code, info, model)
        diagnostics = TrainingDiagnostics(model, ds['train'], cfg, path)
        write_json(path/'summary.json', {'RUN_ID': manifest['RUN_ID'], 'EXP_ID': manifest['EXP_ID'],
                   'trainability': diagnostics.initial['trainability'], 'status': 'COMPLETED', 'optimizer_steps': 0,
                   'purpose': cfg['experiment']['purpose']})
        manifest.update(status='COMPLETED', completed_epoch=0, global_step=0, wall_seconds=time.perf_counter()-start)
        write_json(path/'manifest.json', manifest)
        return path

    @staticmethod
    def _append(path,record):
        with Path(path).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
