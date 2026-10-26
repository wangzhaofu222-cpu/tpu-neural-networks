"""Serial, checkpointed real Lab 0 program; requires E0 and a measured budget first."""
import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import yaml
from src.config import load_config
from src.analysis.lab0_acceptance import e0,e2,repeatability,resume_equivalence,read_run
from src.analysis.statistics import parameter_relative_difference
from src.profile import profile_model
from src.utils.io import write_json,file_hash,digest
from src.utils.provenance import git_state,experiment_id
from src.models.small_cnn import build_model
from src.engine.checkpoint import load_checkpoint
from src.utils.reproducibility import seed_all


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--e0-run',type=Path,required=True)
    parser.add_argument('--e0-evaluation',type=Path,required=True)
    args=parser.parse_args()
    if e0(args.e0_run,args.e0_evaluation)['status']!='PASS':
        raise SystemExit('E0 failed: diagnose before expanding the program')
    budget=json.loads((ROOT/'docs/environment/lab0_budget.json').read_text())
    if not budget['teacher_epoch_target_met']:
        raise SystemExit('CPU epoch budget target failed; no automatic expansion')
    code=git_state(ROOT)
    if code['dirty'] or not code['commit']:
        raise SystemExit('Commit clean source/config before the suite')
    directory=ROOT/'experiments/lab0'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    cfg=load_config(ROOT/'configs/lab0.yaml',['experiment.purpose=formal','runtime.max_seconds=600'])
    base_seed=cfg['profile']['base_seed']
    ledger={'started_unix':time.time(),'budget':budget,'E0':str(args.e0_run),'code_commit':code['commit'],
            'runs':{},'checks':{'E0':e0(args.e0_run,args.e0_evaluation)},
            'status':'RUNNING','serial_training':True}
    def persist():
        write_json(directory/'ledger.json',ledger)
    def execute(key,config=None,extra=None):
        config=config or cfg
        path=directory/(key+'.yaml')
        path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        cmd=[sys.executable,'-m','src.train','--config',str(path),*(extra or [])]
        print(f'START {key}',flush=True)
        last=''
        with (directory/(key+'.stdout.txt')).open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in proc.stdout:
                log.write(line); log.flush()
                last=line.strip()
                print(key+' '+line.rstrip(),flush=True)
            if proc.wait():
                raise RuntimeError(f'{key} failed; no further training scheduled')
        run=Path(last)
        if not run.is_absolute():
            run=ROOT/run
        if not (run/'manifest.json').exists():
            raise RuntimeError(f'{key} did not return a valid run directory')
        ledger['runs'][key]=str(run)
        persist()
        return run
    try:
        persist()
        for mode in ['A','B','C']:
            if mode!='A' and not torch.cuda.is_available():
                continue
            mode_cfg=copy.deepcopy(cfg)
            mode_cfg['runtime']['device']='cpu' if mode=='A' else 'cuda'
            mode_cfg['train']['amp']=mode=='C'
            a=execute(f'E1-{mode}-1',mode_cfg)
            b=execute(f'E1-{mode}-2',mode_cfg)
            ledger['checks'][f'E1-{mode}']=repeatability(a,b,mode)
            continuous=execute(f'E3-{mode}-continuous',mode_cfg)
            interrupted=execute(f'E3-{mode}-interrupted',mode_cfg,['--stop-after-epoch',str(cfg['train']['epochs']//2)])
            resumed=execute(f'E3-{mode}-resumed',mode_cfg,['--resume',str(interrupted/'checkpoints/last.pt')])
            ledger['checks'][f'E3-{mode}']=resume_equivalence(continuous,resumed,mode)
            persist()
        seed_runs=[]
        for offset in range(5):
            seed_cfg=copy.deepcopy(cfg)
            seed_cfg['experiment']['seed']=base_seed+offset
            seed_runs.append(execute(f'E2-{offset}',seed_cfg))
        ledger['checks']['E2']=e2(seed_runs)
        # CF2: worker intervention, explicitly mode C by PDF 9.2.
        worker_cfg=copy.deepcopy(cfg)
        worker_cfg['data']['num_workers']=2
        w1=execute('CF02-workers-1',worker_cfg)
        w2=execute('CF02-workers-2',worker_cfg)
        ledger['checks']['CF02']=repeatability(w1,w2,'C')
        reference=read_run(ledger['runs']['E1-A-1'])
        worker=read_run(w1)
        ledger['checks']['CF02']['against_workers0']={
            'split_equal':reference['manifest']['split_fingerprint']==worker['manifest']['split_fingerprint'],
            'first_batch_equal':reference['manifest']['first_batch_checksum']==worker['manifest']['first_batch_checksum'],
            'step_losses_equal':reference['steps']==worker['steps'],
            'final_accuracy_difference':abs(reference['summary']['validation_metric']-worker['summary']['validation_metric'])}
        # CF3: remove every field except model; retain source files and full checkpoint intact.
        partial=Path(ledger['runs']['E3-A-interrupted'])
        complete_state=torch.load(partial/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        truncated=partial/'checkpoints/model_only.pt'
        torch.save({'model':complete_state['model']},truncated)
        abl=copy.deepcopy(cfg)
        abl['experiment']['resume_ablation']='weights_only'
        weights=execute('CF03-weights-only',abl,['--resume',str(truncated),'--weights-only-epoch',str(cfg['train']['epochs']//2)])
        cont=torch.load(Path(ledger['runs']['E3-A-continuous'])/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        broken=torch.load(weights/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        ledger['checks']['CF03']={'delta_theta':parameter_relative_difference(cont['model'],broken['model']),
            'discarded_fields':[k for k in complete_state if k!='model'],'RUN_ID':weights.name}
        # CF4: search the prescribed seed series; do not invent a near-equal pair if none exists.
        pairs=[]
        for i,a in enumerate(seed_runs):
            for b in seed_runs[i+1:]:
                ra,rb=read_run(a),read_run(b)
                gap=abs(ra['summary']['validation_metric']-rb['summary']['validation_metric'])
                if gap<.001:
                    pairs.append({'RUN_IDs':[a.name,b.name],'accuracy_difference':gap,'trajectories_equal':ra['steps']==rb['steps']})
        ledger['checks']['CF04']={'pairs':pairs,'status':'MEASURED' if pairs else 'NO_QUALIFYING_PAIR; further seeds required'}
        # CF6: one-byte change on a copy, never mutate original downloaded data.
        original=ROOT/'data/cifar-10-batches-py/data_batch_1'
        mutated=ROOT/'tmp'/('data-byte-probe-'+uuid.uuid4().hex+'.bin')
        shutil.copy2(original,mutated)
        with mutated.open('r+b') as stream:
            first=stream.read(1); stream.seek(0); stream.write(bytes([first[0]^1]))
        ledger['checks']['CF06']={'reference_run':reference['manifest']['RUN_ID'],'original_hash':file_hash(original),
            'changed_copy_hash':file_hash(mutated),'detected':file_hash(original)!=file_hash(mutated),
            'original_data_untouched':True,'note':'Official torchvision MD5 would also reject this corrupted copy before training.'}
        # CF5 is a manifest counterfactual, not an actual library upgrade.
        ledger['checks']['CF05']={'reference_run':reference['manifest']['RUN_ID'],
            'cudnn_recorded':reference['manifest']['environment']['cudnn'],
            'environment_in_EXP_ID':False,'actual_library_upgrade_performed':False}
        # E4 and CF7/8: trained CPU model; per-size independent timing, CUDA memory if available.
        trained=Path(ledger['runs']['E1-A-1'])
        for device_name in ['cpu','cuda']:
            if device_name=='cuda' and not torch.cuda.is_available():
                continue
            device=torch.device(device_name)
            seed_all(base_seed,True,2)
            model=build_model(cfg['model']).to(device)
            load_checkpoint(trained/'checkpoints/last.pt',model)
            measurements=profile_model(model,cfg['profile_batch_sizes'],device,50,100)
            measurements.update(source_RUN_ID=trained.name,checkpoint_sha256=file_hash(trained/'checkpoints/last.pt'),
                                status='MEASURED; inspect CV flags; no automatic threshold override')
            write_json(directory/f'E4-{device_name}.json',measurements)
        ledger['status']='TRAINING_PROGRAM_COMPLETED; REPORT_AND_CF_COMPLETENESS_PENDING'
        ledger['wall_seconds']=time.time()-ledger['started_unix']
        persist()
        print('SUITE_DIRECTORY '+str(directory),flush=True)
    except BaseException as exc:
        ledger['status']='FAILED'
        ledger['error']={'type':type(exc).__name__,'message':str(exc)}
        persist()
        raise


if __name__=='__main__':
    main()
