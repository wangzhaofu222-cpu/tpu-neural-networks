"""Check a real one-epoch run after exposing unchanged split settings in config."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); a=p.parse_args()
    reference=ROOT/'runs/6a750eb4531ac754cde7-b798991a1a7b457cbb0de2144b2a4742'
    before,after=read(reference/'manifest.json'),read(a.run/'manifest.json')
    first=json.loads((reference/'metrics.jsonl').read_text(encoding='utf-8').splitlines()[0]); actual=read(a.run/'summary.json')
    checks={key:before[key]==after[key] for key in ['data_fingerprint','split_fingerprint','subset_fingerprint',
        'actual_train_size','normalization','first_batch_checksum','parameters','profile','seed']}
    checks.update(first_epoch_train_loss=first['train_loss']==actual['train_loss'],
                  first_epoch_validation_loss=first['validation_loss']==actual['validation_loss'],
                  first_epoch_accuracy=first['validation_metric']==actual['validation_metric'],
                  one_epoch_only=actual['epoch']==1,real_data=not after['synthetic_fixture'],
                  clean_committed_release=bool(after['git_commit']) and not after['git_dirty'])
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'scope':'Unchanged data pipeline and exact real first-epoch numerical behavior after config-only split exposure and compare/status display changes',
        'full_formal_suite_rerun_after_release_changes':False,'reference_RUN_ID':reference.name,'release_RUN_ID':a.run.name,
        'reference_commit':before['git_commit'],'release_commit':after['git_commit'],
        'train_loss':actual['train_loss'],'validation_accuracy':actual['validation_metric']}
    path=ROOT/'docs/environment/release-compatibility.json'; path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status']!='PASS': raise SystemExit(1)


if __name__=='__main__': main()
