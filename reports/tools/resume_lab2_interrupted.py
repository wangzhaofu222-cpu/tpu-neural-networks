"""Recover the externally interrupted E1 condition without changing its config."""
import json
import copy
import time
from datetime import datetime,timezone
from pathlib import Path
import torch
from src.config import load_config
from src.engine.runner import ExperimentRunner
from src.utils.io import write_json
from src.utils.provenance import git_state
from scripts.lab2_suite import add_record

def main():
    source=Path('runs/lab2/1127217ff6ad88909e85-6dfccf80698847269119e905722501dc')
    audit_path=Path('experiments/lab2/interruption.json')
    if any(r['label']=='E1_cnn_r0.6_j1' for r in json.loads(Path('experiments/lab2/ledger.json').read_text())):
        raise RuntimeError('Condition already completed; refuse duplicate recovery')
    checkpoint=source/'checkpoints/last.pt';st=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert not st['partial_epoch'] and st['global_step']==1010 and st['source_hash']==git_state()['source_hash']
    lines=[json.loads(v) for v in (source/'steps.jsonl').read_text().splitlines()]
    epochs=[json.loads(v) for v in (source/'metrics.jsonl').read_text().splitlines()]
    manifest=json.loads((source/'manifest.json').read_text())
    audit={'source_RUN_ID':source.name,'original_manifest_before_status_update':copy.deepcopy(manifest),
           'observed_not_running_at_utc':datetime.now(timezone.utc).isoformat(),
           'last_recorded_global_step':lines[-1]['global_step'],'checkpoint_global_step':st['global_step'],
           'recorded_updates_replayed':lines[-1]['global_step']-st['global_step'],
           'original_complete_data_epoch_seconds':sum(v['epoch_time'] for v in epochs),
           'original_monotonic_total_seconds':None,
           'timing_limit':'External termination bypassed finally; exact original process total is unavailable. Preserve recorded complete-epoch times, not fabricated total.',
           'recovery':'Same config/source, full optimizer/scheduler/scaler/RNG/loader RNG, complete-data-epoch boundary; new child RUN_ID retains only checkpoint-consistent log prefix.'}
    write_json(audit_path,audit)
    manifest.update(status='INTERRUPTED_EXTERNAL_PROCESS',last_recorded_global_step=lines[-1]['global_step'],
                    last_recoverable_global_step=st['global_step'],interruption_audit=audit_path.as_posix())
    write_json(source/'manifest.json',manifest)
    del st
    cfg=load_config(source/'config.resolved.yaml');start=time.perf_counter()
    path=ExperimentRunner(cfg).run(resume=checkpoint)
    recovered_lines=(path/'steps.jsonl').read_text().splitlines()
    original_lines=(source/'steps.jsonl').read_text().splitlines()
    assert original_lines==recovered_lines[:len(original_lines)]
    add_record('E1_cnn_r0.6_j1',path,time.perf_counter()-start,kind='training',
               resumed_from=source.name,inherited_optimizer_steps=1010,new_optimizer_updates=8990)
    audit.update(recovered_RUN_ID=path.name,recovered_status='COMPLETED',new_optimizer_updates=8990,
                 overlapping_step_records_verified=len(original_lines),overlapping_trajectory_exact=True)
    write_json(audit_path,audit)

if __name__=='__main__':main()
