"""Independent final evidence audit; never mutates historical run records."""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from src.utils.io import digest, file_hash, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git(*args, text=True):
    return subprocess.run(['git','-c','safe.directory=D:/神经网络',*args],capture_output=True,
                          check=True,encoding='utf-8' if text else None).stdout


def main():
    root = Path('reports/lab1')
    acceptance = read(root/'acceptance.json')
    checks = acceptance['gates']
    ledger = read('experiments/lab1/ledger.json')
    prereg = read('experiments/lab1/preregistration_evidence.json')
    commit = prereg['git_commit']
    checks.pop('hypothesis_committed_bytes',None)
    checks.pop('preregistration_committed_bytes',None)
    ending_audit={}
    for field,name,recorded in [('hypothesis','hypothesis.md','hypothesis_sha256'),
                ('preregistration','experiments/lab1/e3_preregistration.json','configuration_sha256')]:
        working=Path(name).read_bytes()
        committed=git('show',commit+':'+name,text=False)
        preserved=hashlib.sha256(working).hexdigest()==prereg[recorded]
        same=working.replace(b'\r\n',b'\n')==committed.replace(b'\r\n',b'\n')
        checks[field+'_committed_content_and_original_working_bytes']=preserved and same
        ending_audit[field]={'original_recorded_working_sha256':prereg[recorded],
                'current_working_sha256':hashlib.sha256(working).hexdigest(),
                'committed_blob_sha256':hashlib.sha256(committed).hexdigest(),
                'original_working_bytes_preserved':preserved,'normalized_content_identical':same,
                'note':'Git core.autocrlf stores LF; Python text output on Windows uses CRLF. Both original working hash and committed content verified.'}
    write_json(root/'preregistration_byte_audit.json',ending_audit)
    e3=[]
    config_pairs = {}
    train_counts_ok = True
    diagnostics_ok = True
    metric_errors=[]
    for r in ledger:
        path=Path(r['path']); manifest=read(path/'manifest.json')
        cfg=__import__('yaml').safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8'))
        if manifest['purpose'] != 'formal':
            continue
        if r['label'].startswith('E3_'):
            git('merge-base','--is-ancestor',commit,manifest['git_commit'])
            assert datetime.fromisoformat(manifest['started_at']) > datetime.fromisoformat(prereg['recorded_at'])
            e3.append(manifest)
            config_pairs[r['label']]=cfg
        if not r['probe_only'] and manifest['status']=='COMPLETED':
            epochs=cfg['train']['epochs']
            steps=sum(1 for _ in (path/'steps.jsonl').open(encoding='utf-8'))
            train_counts_ok &= steps==epochs*336 and manifest['global_step']==epochs*336 and manifest['completed_epoch']==epochs
            data=[json.loads(s) for s in (path/'diagnostics.jsonl').read_text().splitlines()]
            diagnostics_ok &= len(data)==epochs*3
            import numpy as np
            values={}
            for sample in data:
                for layer,row in sample['gradients'].items():
                    if row.get('g') is not None:
                        values.setdefault(layer,[]).append(row['g'])
            medians=np.array([np.median(v) for v in values.values()])
            gi=float(np.log10((medians.max()+1e-12)/(medians.min()+1e-12)))
            final=read(path/'probe.final.json')
            variances=np.array([row['variance'] for row in final['activations'].values()])
            avd=float(np.abs(np.log10(variances+1e-12)-np.log10(variances[0]+1e-12)).max())
            summary=read(path/'summary.json')['trainability']
            metric_errors.append({'RUN_ID':r['RUN_ID'],'GI_error':abs(gi-summary['GI']),'AVD_error':abs(avd-summary['AVD'])})
        protocol=read(path/'diagnostic_protocol.json')
        diagnostics_ok &= len(protocol['probe_indices_in_train_subset'])==64 and len(protocol['loss_indices_in_train_subset'])==1024
        diagnostics_ok &= protocol['probe_mode']=='eval' and cfg['data']['train_fraction']==1.
    checks['preregistered_before_all_E3'] = len(e3)==6
    differences=[]
    from src.compare import diff_configs
    for seed in [1,2,3]:
        diff=diff_configs(config_pairs[f'E3_A_seed{seed}'],config_pairs[f'E3_B_seed{seed}'])
        differences.append(diff)
    checks['E3_exactly_one_scientific_factor'] = all(set(d)=={'experiment.name','model.name'} for d in differences)
    checks['full_epoch_step_counts'] = train_counts_ok
    checks['fixed_probe_and_gradient_sampling'] = diagnostics_ok
    checks['GI_AVD_independent_recomputation'] = all(max(r['GI_error'],r['AVD_error'])<1e-12 for r in metric_errors)
    checks['rho_independent_CPU_checkpoint_evaluation'] = read('docs/environment/lab1-rho-independent.json')['passed']
    write_json(root/'metric_audit.json',metric_errors)
    snapshots=[]
    for p in Path('experiments/lab1').glob('source-*.json'):
        snap=read(p)
        assert digest(snap['files'])==snap['source_hash']
        line_endings={}
        for name,expected in snap['files'].items():
            raw=git('show',snap['git_commit']+':'+name,text=False)
            if hashlib.sha256(raw).hexdigest()!=expected:
                import base64
                original=base64.b64decode(snap['byte_overrides'][name]['base64'])
                assert hashlib.sha256(original).hexdigest()==expected, (name,'original byte hash mismatch')
                assert original.replace(b'\r\n',b'\n')==raw.replace(b'\r\n',b'\n'), (name,'source content differs from committed version')
                line_endings[name]='Exact original bytes archived; mixed LF/CRLF normalized by Git; normalized source content verified'
        snapshots.append({'path':p.as_posix(),'source_hash':snap['source_hash'],'git_commit':snap['git_commit'],'line_endings':line_endings})
    source_hashes={s['source_hash'] for s in snapshots}
    checks['formal_source_snapshots_match_Git'] = all(read(Path(r['path'])/'manifest.json')['source_hash'] in source_hashes for r in ledger if read(Path(r['path'])/'manifest.json')['purpose']=='formal')
    preserved={}
    for name in ['docs/environment/lab0-before-lab1.json','docs/environment/lab0-run-records-before-lab1.json']:
        for p,h in read(name).items():
            preserved[p]=Path(p).is_file() and file_hash(p)==h
    checks['Lab0_all_preserved_artifact_bytes'] = all(preserved.values())
    suite=ET.parse('docs/environment/tests-lab1-final.xml').getroot().find('testsuite').attrib
    checks['all_automated_tests_pass'] = all(int(suite[k])==0 for k in ['failures','errors','skipped'])
    build=read(root/'build_status.json')
    checks['LaTeX_compiled'] = build['latex_compile_status']=='PASS'
    checks['all_PDF_pages_visually_verified'] = build['visual_QA_status']=='PASS' and build['pages_inspected']==list(range(1,build['pages']+1))
    checks['PDF_hash_matches_QA'] = file_hash(root/'report.pdf')==build['pdf_sha256']
    assert all((root/f'images/{n}.png').is_file() for n in ['e1','e2','map'])
    complete = all(v for k,v in checks.items() if k!='E2_growth_thresholds')
    status = 'PASS' if all(checks.values()) else 'COMPLETE_WITH_EXPLAINED_THRESHOLD_FAILURE' if complete else 'FAIL'
    acceptance.update(gates=checks,pending=[],status=status,teacher_completion_standard_pass=complete,
        all_numeric_thresholds_pass=checks['E1_chi_within_20_percent'] and checks['E2_growth_thresholds'],
        tests=suite,Lab0_preserved_files=len(preserved),source_snapshots=snapshots,E3_config_differences=differences,
        final_audit_commit=git('rev-parse','HEAD').strip())
    write_json(root/'acceptance.json',acceptance)
    write_json(root/'preservation_audit.json',{'files_checked':len(preserved),'all_identical':all(preserved.values()),
             'mismatches':[p for p,ok in preserved.items() if not ok]})
    print(json.dumps({'status':acceptance['status'],'checks':checks,'tests':suite,'Lab0_files':len(preserved)},indent=2))
    if not complete:
        raise SystemExit(1)


if __name__=='__main__':
    main()
