"""Read-only acceptance audit over immutable real run artifacts."""
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.analysis.lab0_acceptance import e0,e2,repeatability,resume_equivalence,read_run
from src.utils.io import write_json


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--suite',type=Path,required=True); p.add_argument('--report',type=Path,required=True)
    a=p.parse_args(); ledger=read(a.suite/'ledger.json'); checks={}; selected=read(ROOT/'docs/environment/e0-selected.json')
    checks['E0']=e0(ROOT/'runs'/selected['RUN_ID'],ROOT/selected['evaluation_path'])['status']=='PASS'
    checks['suite_finished']=ledger['status'].startswith('TRAINING_PROGRAM_COMPLETED')
    for mode in ['A','B','C']:
        runs=ledger['runs']; checks[f'E1-{mode}']=repeatability(runs[f'E1-{mode}-1'],runs[f'E1-{mode}-2'],mode)['status']=='PASS'
        x,y=runs[f'E3-{mode}-continuous'],runs[f'E3-{mode}-resumed']
        checks[f'E3-{mode}']=resume_equivalence(x,y,mode)['status']=='PASS'
        checks[f'E3-{mode}-exact-step-losses']=read_run(x)['steps']==read_run(y)['steps']
    stats=e2([ledger['runs'][f'E2-{i}'] for i in range(5)])
    checks['E2']=stats['n']==5 and stats['ci95'][0]<stats['mean']<stats['ci95'][1]
    checks['CF02']=repeatability(ledger['runs']['CF02-workers-1'],ledger['runs']['CF02-workers-2'],'C')['status']=='PASS'
    checks['CF03_nonzero_ablation']=ledger['checks']['CF03']['delta_theta']>0
    extra=read(a.suite/'CF04-extra.json') if (a.suite/'CF04-extra.json').exists() else {}
    pairs=extra.get('pairs',[]) or ledger['checks']['CF04']['pairs']
    checks['CF04']=bool(pairs)
    for pair in pairs:
        aa,bb=[read_run(ROOT/'runs'/x) for x in pair['RUN_IDs']]
        assert aa['manifest']['seed']!=bb['manifest']['seed']
        assert abs(aa['summary']['validation_metric']-bb['summary']['validation_metric'])<.001
        assert pair['trajectories_equal']==(aa['steps']==bb['steps'])
    checks['CF06']=read(ROOT/'docs/environment/data-byte-probe.json')['official_data_untouched']
    for device in ['cpu','cuda']:
        data=read(a.suite/f'E4-{device}-registered.json'); rows=data['measurements']
        checks[f'E4-{device}']=([r['batch_size'] for r in rows]==[1,16,64,256] and
            all(len(r['samples_seconds'])==r['repeats'] and r['warmup']>=10 and r['repeats']>=100 for r in rows))
        checks[f'E4-{device}-primary-CV']=all(r['cv_acceptable'] for r in rows)
        assert (ROOT/'runs'/data['RUN_ID']/'manifest.json').exists()
    text=(a.report/'report.ru.md').read_text(encoding='utf-8')
    checks['report_six_sections']=len(re.findall(r'^## [1-6]\.',text,re.M))==6
    checks['report_eight_CF']=len(re.findall(r'^[1-8]\. ',text.split('## 5.')[1].split('## 6.')[0],re.M))==8
    images=re.findall(r'!\[.*?\]\((.*?)\)',text)
    checks['report_images']=len(images)>=7 and all((a.report/x).exists() for x in images)
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'E2':stats,
            'scope':'Lab0 experimental evidence and six-section source report only; PDF QA and code/config audit recorded separately',
            'course_Lab1_to_9_complete':False,'suite':str(a.suite),'tested_commit':ledger['code_commit']}
    if (a.suite/'E4-cuda-warmed.json').exists():
        result['extended_GPU_timing']={'CV_threshold_status':'FAIL','repeats':300,
            'batches_above_CV_10_percent':[r['batch_size'] for r in read(a.suite/'E4-cuda-warmed.json')['measurements'] if not r['cv_acceptable']],
            'interpretation':'Additional diagnostic retained alongside passing primary N100 series; global latency stability is not established'}
    write_json(a.report/'acceptance.json',result); print(json.dumps(result,indent=2))
    if result['status']!='PASS': raise SystemExit(1)


if __name__=='__main__': main()
