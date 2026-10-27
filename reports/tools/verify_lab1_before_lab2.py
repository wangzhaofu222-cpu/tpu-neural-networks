"""Historical evidence audit; new outputs are isolated from Lab0/1 artifacts."""
import json, math, statistics, subprocess
from pathlib import Path
import torch, yaml
from datetime import datetime
from src.compare import diff_configs
from src.engine.runner import ExperimentRunner
from src.utils.io import digest, file_hash, write_json

OUT=Path('experiments/lab1_verification')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    OUT.mkdir(exist_ok=True)
    roots=[Path(p) for p in ['reports/lab0','reports/lab1','experiments/lab0','experiments/lab1']]
    roots += [p for p in Path('runs').iterdir() if p.is_dir()]
    before={p.as_posix():file_hash(p) for root in roots for p in root.rglob('*') if p.is_file()}
    write_json(OUT/'historical_files.json',before)
    ledger=read('experiments/lab1/ledger.json');runs={r['label']:r for r in ledger}
    checkpoints=[];torch.set_num_threads(2)
    for row in ledger:
        p=Path(row['path']);m=read(p/'manifest.json');s=read(p/'summary.json')
        cfg=yaml.safe_load((p/'config.resolved.yaml').read_text(encoding='utf-8'))
        assert m['RUN_ID']==p.name and m['EXP_ID']==p.name.split('-')[0]
        assert digest(cfg)==m['config_hash']
        assert (p/'probe.initial.json').is_file() and (p/'diagnostic_protocol.json').is_file()
        if row['probe_only']:
            assert m['global_step']==0
        elif m['status']=='COMPLETED':
            assert (p/'metrics.jsonl').is_file() and (p/'steps.jsonl').is_file()
            for kind in ['last','best']:
                path=p/f'checkpoints/{kind}.pt';assert path.is_file()
                st=torch.load(path,map_location='cpu',weights_only=False)
                assert st['run_id']==m['RUN_ID'] and st['source_hash']==m['source_hash']
                assert st['config_hash']==m['config_hash']
                assert all(k in st for k in ['optimizer','scheduler','scaler','torch_rng','loader_rng'])
                assert all(torch.isfinite(v).all() for v in st['model'].values() if v.is_floating_point())
                if kind=='last': assert st['global_step']==m['global_step'] and st['epoch']==m['completed_epoch']
                checkpoints.append({'RUN_ID':m['RUN_ID'],'kind':kind,'sha256':file_hash(path),'epoch':st['epoch'],'steps':st['global_step']})
                del st
        else:
            assert m['status']=='FAILED' and s['trainability']['numerical_failure']
            assert (p/'failure.json').is_file() and m['completed_epoch']==0
    original=runs['E2_residual_scaled_96'];diag=runs['E2_scaled96_growth_diagnosis']
    initial=read(Path(original['path'])/'probe.initial.json');detailed=read(Path(diag['path'])/'probe.initial.json')
    rows=list(detailed['branch_moments'].values());c0=rows[0]['branch_q']/rows[0]['input_q']
    actual=initial['activations']['h96']['q']/initial['activations']['h0']['q']
    product=math.prod(1+r['c']/96+2*r['normalized_cross']/math.sqrt(96) for r in rows)
    assert math.isclose(c0,initial['c'],rel_tol=1e-14) and math.isclose(actual,initial['q_ratio'],rel_tol=1e-14)
    assert math.isclose(product,actual,rel_tol=1e-12)
    cfg=yaml.safe_load((Path(original['path'])/'config.resolved.yaml').read_text(encoding='utf-8'))
    cfg['experiment'].update(name='Lab1_E2_verification_replay',purpose='audit')
    cfg['logging']['run_root']='runs/lab1_verification'
    replay=ExperimentRunner(cfg).run_probe();fresh=read(replay/'probe.initial.json')
    assert math.isclose(fresh['q_ratio'],actual,rel_tol=1e-12) and math.isclose(fresh['c'],c0,rel_tol=1e-12)
    A=[];B=[];ids=[]
    differences=[]
    for seed in [1,2,3]:
        pair=[];configs=[]
        for arm,values in [('A',A),('B',B)]:
            r=runs[f'E3_{arm}_seed{seed}'];p=Path(r['path']);m=read(p/'manifest.json')
            assert m['seed']==seed and not m['git_dirty']
            values.append(read(p/'summary.json')['trainability']['rho_T']);pair.append(r['RUN_ID'])
            configs.append(yaml.safe_load((p/'config.resolved.yaml').read_text(encoding='utf-8')))
        ids.append(pair)
        diff=diff_configs(*configs);assert set(diff)=={'experiment.name','model.name'};differences.append(diff)
    diffs=[a-b for a,b in zip(A,B)];mean=statistics.mean(diffs);sd=statistics.stdev(diffs)
    registration=read('experiments/lab1/preregistration_evidence.json');commit=registration['git_commit']
    hyp=subprocess.run(['git','-c','safe.directory=D:/神经网络','show',commit+':hypothesis.md'],capture_output=True,check=True).stdout
    assert hyp.replace(b'\r\n',b'\n')==Path('hypothesis.md').read_bytes().replace(b'\r\n',b'\n')
    for pair in ids:
        for run in pair:
            m=read(Path('runs/lab1')/run/'manifest.json')
            subprocess.run(['git','-c','safe.directory=D:/神经网络','merge-base','--is-ancestor',commit,m['git_commit']],check=True)
            assert datetime.fromisoformat(m['started_at'])>datetime.fromisoformat(registration['recorded_at'])
    e1=[]
    for init,expected in [('he',1.),('xavier',.5),('gaussian',.0512)]:
        for depth in [8,24,48,96]:
            rr=runs[f'E1_{init}_{depth}'];data=read(Path(rr['path'])/'probe.initial.json')
            assert all(all(k in row for k in ['mean','std','variance','q','min','max','delta']) for row in data['activations'].values())
            if depth==96: assert all(abs(data['chi'][k]/expected-1)<=.2 for k in ['fwd','bwd'])
            e1.append({'RUN_ID':rr['RUN_ID'],'chi':data['chi'],'predicted':expected})
    assert all(file_hash(p)==h for p,h in before.items())
    write_json(OUT/'results.json',{'historical_files_preserved':len(before),'code_error_found':False,'training_repeated':False,
      'fresh_forward_probe_RUN_ID':replay.name,'E2':{'RUN_ID':original['RUN_ID'],'diagnostic_RUN_ID':diag['RUN_ID'],
       'c_first':c0,'mean_c':statistics.mean(r['c'] for r in rows),'measured_ratio':actual,'bound':1.2*math.exp(c0),
       'excess_fraction':actual/(1.2*math.exp(c0))-1,'exact_moment_product':product,'status':'FAIL_RETAINED',
       'cause':'First-block c is not constant across finite-width branches; empirical cross terms are nonzero. Exact per-block identity reconstructs q. No seed/threshold changes.'},
      'E3':{'A':A,'B':B,'paired_RUN_IDs':ids,'deltas':diffs,'mean':mean,'sample_std':sd,
       'established':all(d>0 for d in diffs) and mean>=2*sd,'all_B_trainable':all(v<=.5 for v in B),'preregistration_commit':commit},
      'E1':e1,'E3_config_differences':differences,'checkpoint_checks':checkpoints,'status':'VERIFIED_WITH_EXISTING_E2_THRESHOLD_FAILURE'})
    print(json.dumps({'historical_files':len(before),'checkpoints_verified':len(checkpoints),'E2':'FAIL retained; replay matches','E3_deltas':diffs,'replay_RUN_ID':replay.name},indent=2))

if __name__=='__main__': main()
