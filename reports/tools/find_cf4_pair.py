"""Optional extra seeds for CF4; never alter the preplanned five-run E2 statistics."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import yaml
from src.config import load_config
from src.analysis.lab0_acceptance import read_run
from src.utils.io import write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument('--suite',type=Path,required=True); p.add_argument('--max-extra',type=int,default=4)
    a=p.parse_args(); ledger=json.loads((a.suite/'ledger.json').read_text(encoding='utf-8'))
    paths=[Path(ledger['runs'][f'E2-{i}']) for i in range(5)]
    cfg=load_config(paths[0]/'config.resolved.yaml'); base=cfg['profile']['base_seed']; started=time.time()
    result={'runs':[],'pairs':[],'status':'SEARCHING','primary_E2_N_remains':5,'max_extra':a.max_extra}
    def search():
        runs=[read_run(r) for r in paths]; found=[]
        for i,aa in enumerate(runs):
            for bb in runs[i+1:]:
                gap=abs(aa['summary']['validation_metric']-bb['summary']['validation_metric'])
                if gap<.001:
                    found.append({'RUN_IDs':[aa['path'].name,bb['path'].name],'accuracy_difference':gap,
                                  'trajectories_equal':aa['steps']==bb['steps'],
                                  'max_relative_step_loss_difference':max(abs(x-y)/max(1,abs(x)) for x,y in zip(aa['steps'],bb['steps'])),
                                  'unequal_step_losses':sum(x!=y for x,y in zip(aa['steps'],bb['steps']))})
        return found
    for index in range(a.max_extra+1):
        result['pairs']=search()
        if result['pairs'] or index==a.max_extra: break
        cfg['experiment']['seed']=base+5+index
        config_path=a.suite/f'CF04-extra-{index+1}.yaml'; config_path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        proc=subprocess.Popen([sys.executable,'-m','src.train','--config',str(config_path)],stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',cwd=ROOT)
        last=''
        with (a.suite/f'CF04-extra-{index+1}.stdout.txt').open('w',encoding='utf-8') as log:
            for line in proc.stdout:
                last=line.strip(); log.write(line); log.flush(); print(line.rstrip(),flush=True)
        if proc.wait(): raise RuntimeError('Extra CF4 training failed; original runs preserved')
        path=Path(last); path=path if path.is_absolute() else ROOT/path
        paths.append(path); result['runs'].append(str(path)); write_json(a.suite/'CF04-extra.json',result)
    result['status']='PAIR_FOUND' if result['pairs'] else 'NO_PAIR_WITHIN_EXPLICIT_EXTRA_BUDGET'
    result['wall_seconds']=time.time()-started; write_json(a.suite/'CF04-extra.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
