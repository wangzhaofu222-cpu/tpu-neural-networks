"""Exactly one real CPU epoch before scheduling any full series. Never marks E0 complete."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config import load_config
from src.engine.runner import ExperimentRunner
from src.utils.io import write_json


def main():
    cfg=load_config('configs/lab0.yaml',['experiment.purpose=budget','runtime.max_seconds=180'])
    path=ExperimentRunner(cfg).run(stop_after_epoch=1)
    record=json.loads((path/'summary.json').read_text())
    seconds=record['epoch_time']
    result={'RUN_ID':path.name,'epoch_seconds':seconds,'epochs_planned':cfg['train']['epochs'],
        'mandatory_13_runs_estimated_seconds':13*cfg['train']['epochs']*seconds,
        'cuda_additional_runs':'NOT MEASURED; up to 8 runs',
        'teacher_epoch_target_met':seconds<=60,
        'status':'BUDGET_ONLY_NOT_E0','cpu_threads':cfg['runtime']['threads'],
        'cpu_temperature':'not_measured; user requested not to wait for temperature monitoring',
        'automatic_full_series_started':False}
    write_json('docs/environment/lab0_budget.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
