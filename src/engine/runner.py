import json
import math
import shutil
import time
from pathlib import Path
import torch
import torch.nn.functional as F
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

    def run(self,resume=None,stop_after_epoch=None,weights_only_epoch=None):
        cfg = self.config
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
                     torch.optim.SGD(model.parameters(),momentum=opt_cfg['momentum'],**options))
        total_epochs = cfg['train']['epochs']
        warmup = cfg['scheduler']['warmup_epochs']
        def multiplier(epoch):
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
            if state['config_hash'] != digest(cfg) or state['data_fingerprint'] != data_info['data_fingerprint'] or state['split_fingerprint'] != data_info['split_fingerprint']:
                raise ValueError('Resume config/data/split mismatch')
            if state['source_hash'] != code['source_hash']:
                raise ValueError('Resume source changed')
            start_epoch,global_step = state['epoch'],state['global_step']
            best,best_epoch,parent = state['best_accuracy'],state['best_epoch'],state['run_id']
        path,manifest = create_run(cfg,code,data_info,model,parent)
        (path/'checkpoints').mkdir()
        if resume:
            parent_path = Path(resume).parent.parent
            for name in ['metrics.jsonl','steps.jsonl']:
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
        completed_epoch = start_epoch
        status = 'RUNNING'
        try:
            for epoch in range(start_epoch,total_epochs):
                epoch_start = time.perf_counter()
                model.train()
                loss_sum,correct,count = 0.,0,0
                lr_used = optimizer.param_groups[0]['lr']
                epoch_initial_step = global_step
                optimizer.zero_grad(set_to_none=True)
                accumulation = cfg['train']['accumulation_steps']
                for batch_index,(x,y) in enumerate(dl['train']):
                    optimizer_step_skipped = False
                    if budget and time.perf_counter()-wall_start > budget:
                        raise TimeoutError('Configured wall time exceeded; no incomplete-epoch checkpoint claimed')
                    if first_checksum is None:
                        first_checksum = digest({'images':x.tolist(),'labels':y.tolist()})
                    x,y = x.to(device),y.to(device)
                    if diagnostics:
                        diagnostics.begin(batch_index, len(dl['train']))
                    with torch.autocast(device_type=device.type,enabled=cfg['train']['amp']):
                        logits = model(x)
                        loss = F.cross_entropy(logits,y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    # Normalize the final, possibly shorter accumulation group correctly.
                    group_start = (batch_index//accumulation)*accumulation
                    group_length = min(accumulation,len(dl['train'])-group_start)
                    group_examples = sum(min(cfg['train']['batch_size'],len(ds['train'])-j*cfg['train']['batch_size'])
                                         for j in range(group_start,group_start+group_length))
                    scaled_loss = loss if accumulation == 1 else loss*len(y)/group_examples
                    scaler.scale(scaled_loss).backward()
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
                    loss_sum += loss.item()*len(y)
                    correct += (logits.argmax(1)==y).sum().item()
                    count += len(y)
                    self._append(path/'steps.jsonl',{'epoch':epoch+1,'batch':batch_index,'global_step':global_step,'train_loss':loss.item(),'learning_rate':lr_used,'optimizer_step_skipped':optimizer_step_skipped})
                val = evaluate_loader(model,dl['val'],device)
                if diagnostics and not math.isfinite(val['loss']):
                    raise FloatingPointError('Non-finite validation loss')
                if global_step > epoch_initial_step:
                    scheduler.step()
                completed_epoch = epoch+1
                record = {'epoch':completed_epoch,'global_step':global_step,'train_loss':loss_sum/count,
                          'validation_loss':val['loss'],'train_metric':correct/count,'validation_metric':val['accuracy'],
                          'learning_rate':lr_used,'epoch_time':time.perf_counter()-epoch_start,
                          'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
                self._append(path/'metrics.jsonl',record)
                improved = val['accuracy'] > best
                if improved:
                    best,best_epoch = val['accuracy'],completed_epoch
                extra = {'best_accuracy':best,'best_epoch':best_epoch,'run_id':manifest['RUN_ID'],
                         'config_hash':digest(cfg),'data_fingerprint':data_info['data_fingerprint'],
                         'split_fingerprint':data_info['split_fingerprint'],'source_hash':code['source_hash'],
                         'first_batch_checksum':first_checksum}
                if completed_epoch%cfg['logging']['save_every']==0 or completed_epoch==total_epochs or completed_epoch==stop_after_epoch:
                    save_checkpoint(path/'checkpoints/last.pt',model,optimizer,scheduler,scaler,completed_epoch,global_step,g,**extra)
                if improved:
                    save_checkpoint(path/'checkpoints/best.pt',model,optimizer,scheduler,scaler,completed_epoch,global_step,g,**extra)
                write_json(path/'summary.json',{**record,'RUN_ID':manifest['RUN_ID'],'EXP_ID':manifest['EXP_ID'],
                           'best_accuracy':best,'best_epoch':best_epoch,'first_batch_checksum':first_checksum,
                           'checkpoint_for_final_metric':'checkpoints/last.pt','purpose':cfg['experiment']['purpose']})
                print(json.dumps({'RUN_ID':manifest['RUN_ID'],**record}),flush=True)
                if stop_after_epoch and completed_epoch >= stop_after_epoch and completed_epoch<total_epochs:
                    status = 'INTERRUPTED_AT_EPOCH_BOUNDARY'
                    break
            else:
                status = 'COMPLETED'
        except Exception as exc:
            status = 'FAILED'
            write_json(path/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed_epoch':completed_epoch})
            if not (diagnostics and isinstance(exc, FloatingPointError)):
                raise
        finally:
            if diagnostics:
                existing = json.loads((path/'summary.json').read_text(encoding='utf-8')) if (path/'summary.json').exists() else {
                    'RUN_ID': manifest['RUN_ID'], 'EXP_ID': manifest['EXP_ID'], 'purpose': cfg['experiment']['purpose']}
                existing['trainability'] = diagnostics.finish(failed=status == 'FAILED')
                write_json(path/'summary.json', existing)
            manifest.update(status=status,completed_epoch=completed_epoch,global_step=global_step,
                            first_batch_checksum=first_checksum,wall_seconds=time.perf_counter()-wall_start)
            write_json(path/'manifest.json',manifest)
            if (path/'summary.json').exists():
                summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
                summary['status']=status
                write_json(path/'summary.json',summary)
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
