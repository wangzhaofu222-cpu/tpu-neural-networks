import copy,subprocess
from datetime import datetime
from pathlib import Path
import torch
from src.utils.io import write_json,file_hash
from src.utils.preregistration import verify_git_hypothesis
from reports.tools.lab7_common import R,base,read,registry,train,evaluate


def main():
    torch.set_num_threads(2)
    if (R/'E3.json').exists():return
    lock=read(R/'E3_protocol_lock.json')
    commit=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','log','-1','--format=%H','--','experiments/lab7/hypothesis.md']).decode().strip()
    assert verify_git_hypothesis(commit,'experiments/lab7/hypothesis.md')
    blob=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show',commit+':experiments/lab7/E3_protocol_lock.json'])
    assert blob.replace(b'\r\n',b'\n')==(R/'E3_protocol_lock.json').read_bytes().replace(b'\r\n',b'\n')
    assert file_hash(R/'E1.json')==lock['E1_sha256'] and file_hash(R/'E2.json')==lock['E2_sha256']
    timestamp=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show','-s','--format=%cI',commit]).decode().strip()
    write_json(R/'preregistration_evidence.json',{'commit':commit,'committed_at':timestamp,'hypothesis_sha256':file_hash(R/'hypothesis.md'),'lock_sha256':file_hash(R/'E3_protocol_lock.json')})
    pairs=[]
    for seed in [1,2,3]:
        cfg=base();cfg['experiment']['seed']=seed
        source=Path(registry()['training']['E1_linear']) if seed==1 else train('E3_linear_'+str(seed),cfg)
        pair={'seed':seed,'training_source':str(source),'seed1_exploratory_reuse':seed==1}
        for arm in ['A','B']:
            a=lock[arm]
            if seed==1:
                rid=lock['seed1_reuse'][arm+'_RUN_ID'];r=next(r for r in registry()['evaluations'].values() if r['RUN_ID']==rid)
            else:r=evaluate('E3_'+arm+'_'+str(seed),source,a['sampler'],a['K'],initial_seed=7400+seed,noise_seed=7500+seed)
            pair[arm]=r
        pairs.append(pair)
    delta=torch.tensor([p['A']['Dmode']-p['B']['Dmode'] for p in pairs],dtype=torch.float64)
    mean=float(delta.mean());sd=float(delta.std());same=bool((delta>0).all() or (delta<0).all())
    established=same and abs(mean)>=2*sd
    noise=read(R/'metrics'/f"{base()['data']['pi0']}.json")
    result={'A':lock['A'],'B':lock['B'],'pairs':pairs,'deltas':delta.tolist(),'mean':mean,'sample_sd':sd,
        'same_sign':same,'abs_mean_ge_2sd':abs(mean)>=2*sd,'Dmode_noise_mean':noise['Dmode_noise_mean'],
        'above_noise':abs(mean)>=noise['Dmode_noise_mean'],'teacher_conclusion':'различие установлено' if established else 'различие не установлено',
        'hypothesis_supported':established and all(float(d)*lock['expected_sign']>0 for d in delta),
        'preregistration_commit':commit}
    write_json(R/'E3.json',result);print('E3 COMPLETE',result['teacher_conclusion'],flush=True)


if __name__=='__main__':main()
