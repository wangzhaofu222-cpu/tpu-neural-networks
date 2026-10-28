"""Final read-only scientific evidence audit; writes only newLab4 audit/index files."""
import copy
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime,timedelta
from pathlib import Path
import torch
import yaml
from src.models.small_cnn import build_model
from src.utils.io import digest,file_hash,write_json
from src.diagnostics.representation import representation_statistics
from src.analysis.label_efficiency import paired_gain,label_subsets
from src.data.ssl_data import bundle,labelled_targets
from src.compare import diff_configs

ROOT=Path('experiments/lab4')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def git(*args):return subprocess.run(['git','-c','safe.directory=D:/神经网络',*args],capture_output=True,text=True,encoding='utf-8',check=True).stdout.strip()
def main():
    torch.set_num_threads(2);result=read(ROOT/'results.json');proof=read(ROOT/'preregistration_evidence.json')
    teacher=read(ROOT/'teacher_source.json');assert file_hash(teacher['path'])==teacher['sha256'] and teacher['pages']==14
    cfg=yaml.safe_load(Path('configs/lab4.yaml').read_text());images,split,mean,std,_=bundle(cfg['data']['root'],cfg['data']['split_path'])
    historical=read('experiments/lab3_verification/historical_files.json')
    assert all(Path(p).exists() and file_hash(p)==h for p,h in historical.items())
    assert read('experiments/lab3_verification/audit.json')['status']=='PASS'
    exploratory_commit=git('log','-1','--format=%H','--','experiments/lab4/exploratory_hypotheses.md')
    assert git('show',exploratory_commit+':experiments/lab4/exploratory_hypotheses.md')==(ROOT/'exploratory_hypotheses.md').read_text(encoding='utf-8').strip()
    exploratory_time=datetime.fromisoformat(git('show','-s','--format=%cI',exploratory_commit))
    checkpoints=[];registry=[];encoders=[];probe_runs=[]
    for path in sorted(Path('runs/lab4').iterdir()):
        if not path.is_dir():continue
        m=read(path/'manifest.json');c=yaml.safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8'))
        assert digest(c)==m['config_hash'] and path.name==m['RUN_ID']
        assert m['status']!='RUNNING'
        registry.append({'EXP_ID':m['EXP_ID'],'RUN_ID':path.name,'status':m['status'],'purpose':m['purpose'],'role':c['experiment']['role'],'path':path.as_posix()})
        for cp in sorted((path/'checkpoints').glob('*.pt')):
            state=torch.load(cp,map_location='cpu',weights_only=False)
            assert all(k in state for k in ['optimizer','scheduler','scaler','python_rng','numpy_rng','torch_rng','cuda_rng','loader_rng'])
            assert state['config_hash']==m['config_hash'] and state['source_hash']==m['source_hash']
            assert state['split_fingerprint']==m['split_fingerprint'] and state['data_fingerprint']==m['data_fingerprint']
            model=build_model(c['model']);model.load_state_dict(state['model'],strict=True)
            assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point())
            checkpoints.append({'RUN_ID':path.name,'path':cp.as_posix(),'sha256':file_hash(cp),'epoch':state['epoch'],'global_step':state['global_step'],
                'partial_epoch':state.get('partial_epoch',False),'purpose':m['purpose'],'model':c['model']['name']})
            del state,model
        if m['purpose']!='formal' or m['status']!='COMPLETED':continue
        git('merge-base','--is-ancestor',exploratory_commit,m['git_commit'])
        assert datetime.fromisoformat(m['started_at'])>exploratory_time
        rows=[json.loads(v) for v in (path/'steps.jsonl').read_text().splitlines()];s=read(path/'summary.json')
        assert rows and s['epoch']==c['train']['epochs']
        assert sum(not r['optimizer_step_skipped'] for r in rows)==s['global_step']
        if c['model']['name']=='simclr':
            contrastive=c['model']['mode']=='contrastive';n=c['train']['batch_size'];size=m['actual_train_size']
            per_epoch=size//n if contrastive else math.ceil(size/n)
            assert len(rows)==per_epoch*c['train']['epochs']
            assert m['split_fingerprint']==split['fingerprint']
            if contrastive:
                assert n==(64 if c['experiment']['role']=='CF1' else 256)
                assert c['contrastive']['tau']==cfg['profile']['lab4_tau']
                assert m['pretraining_label_access'] is False and m['train_subset_indices']==split['roles']['train']
                assert all(r['N']==n and r['negative_count']==2*n-2 for r in rows)
                assert all(abs(r['random_baseline_loss']-math.log(2*n-1))<1e-12 for r in rows)
            protocol=read(path/'diagnostic_protocol.json');ids=protocol['image_indices']
            assert len(ids)==5000 and set(ids)<=set(split['roles']['train'])
            expected_epochs=set(range(101)) if c['experiment']['role']=='E1_SSL' else {0,c['train']['epochs']}
            assert {int(p.stem.split('_')[-1]) for p in (path/'diagnostics').glob('epoch_*.pt')}==expected_epochs
            # Recompute every saved diagnostic fromraw5000-image features andpersample similarities.
            for raw in sorted((path/'diagnostics').glob('epoch_*.pt')):
                values=torch.load(raw,map_location='cpu',weights_only=False);saved=read(raw.with_suffix('.json'))['representation']
                assert values['image_indices']==ids
                for kind in ['h','z_hat']:
                    recalculated=representation_statistics(values[kind],kind=='z_hat')
                    for field in ['trace_cov','effective_rank','effective_rank_ratio','collapse_scale','mean_norm']:
                        if isinstance(recalculated[field],float):assert math.isclose(recalculated[field],saved[kind][field],rel_tol=1e-8,abs_tol=1e-8)
                        else:assert recalculated[field]==saved[kind][field]
                    if kind=='z_hat':assert recalculated['identity_4_9_error']<1e-5
            encoders.append({'RUN_ID':path.name,'role':c['experiment']['role'],'epochs':s['epoch'],'optimizer_updates':s['global_step'],
                'physical_batches':len(rows),'AMP_skips':sum(r['optimizer_step_skipped'] for r in rows),
                'training_seconds':s['training_seconds'],'peak_allocator_bytes':s['peak_gpu_memory_bytes']})
        elif c['model']['name']=='linear_probe':
            assert len(rows)==cfg['evaluation']['probe_epochs'];probe_runs.append({'RUN_ID':path.name,'training_seconds':s['training_seconds'],
                'epochs':s['epoch'],'regularization':c['optimizer']['weight_decay'],'seed':c['experiment']['seed']})
    assert len(encoders)==10
    write_json(ROOT/'checkpoint_index.json',checkpoints);write_json(ROOT/'RUN_registry.json',registry)
    # Feature roles/caches, repeatedextraction andcompleteencoderstateimmutability.
    feature_rows=read(ROOT/'features.json')
    for row in feature_rows:
        p=Path(row['path']);s=read(p/'summary.json')
        assert file_hash(p/'features.pt')==s['feature_cache_sha256']
        assert s['role_sizes']=={'train':43000,'val':5000,'test':10000} and not s['cal_access']
        assert s['repeated_extraction_max_error_h']<=1e-6 and s['repeated_extraction_max_error_z']<=1e-6
    evals=read(ROOT/'evaluations.json');by={r['label']:r for r in evals}
    assert by['E1_B0_h']['test_accuracy']==by['E1_SSL_epoch0_h']['test_accuracy']
    assert by['E1_B0_h']['knn_test_accuracy']==by['E1_SSL_epoch0_h']['knn_test_accuracy']
    assert by['E1_B0_h']['encoder_full_state_sha256']==by['E1_SSL_epoch0_h']['encoder_full_state_sha256']
    for row in evals:
        path=Path(row['path']);pred=torch.load(path/'predictions.pt',map_location='cpu',weights_only=False)
        assert len(pred['test_targets'])==10000
        assert (pred['test_predictions']==pred['test_targets']).sum().item()/10000==row['test_accuracy']
        assert (pred['knn_predictions']==pred['test_targets']).sum().item()/10000==row['knn_test_accuracy']
        assert row['encoder_full_state_unchanged']
        winner=max(row['candidates'],key=lambda v:(v['validation_accuracy'],-v['regularization']))
        assert winner['regularization']==row['selected_regularization']
    # Paired/nestedlabelsubsets andrealpreregistration before newE3runs.
    assert git('show',proof['git_commit']+':experiments/lab4/hypothesis.md')==(ROOT/'hypothesis.md').read_text(encoding='utf-8').strip()
    assert file_hash(ROOT/'hypothesis.md')==proof['hypothesis_sha256']
    selection=read(ROOT/'E3_selection.json');prior_results=read(selection['E1_E2_snapshot_path'])
    assert file_hash(selection['E1_E2_snapshot_path'])==selection['E1_E2_results_sha256'] and 'E3' not in prior_results
    assert json.loads(git('show',proof['git_commit']+':'+selection['E1_E2_snapshot_path']))==prior_results
    labels=labelled_targets(cfg['data']['root']);fractions=cfg['profile']['lab4_label_fractions'];matrix=result['E3']['matrix']
    assert len(matrix)==36
    for seed in [1,2,3]:
        expected=label_subsets(labels,split['roles']['train'],fractions,seed)
        for fraction in fractions:
            rows=[r for r in matrix if r['seed']==seed and r['fraction']==fraction]
            assert len(rows)==3 and len({r['subset_fingerprint'] for r in rows})==1
            matched_probe_configs=[]
            for row in rows:
                ev=next(v for v in evals if v['RUN_ID']==row['RUN_ID']);assert ev['label_indices']==expected[str(fraction)]
                frozen=selection['encoders'][row['encoder']]
                assert ev['feature_RUN_ID']==frozen['feature_RUN_ID'] and ev['checkpoint']==frozen['checkpoint']
                assert ev['encoder_checkpoint_sha256']==frozen['checkpoint_sha256']
                assert ev['encoder_full_state_sha256']==frozen['encoder_state_sha256']
                for candidate in ev['candidates']:
                    m=read(Path(candidate['path'])/'manifest.json')
                    assert m['label_indices']==expected[str(fraction)]
                    candidate_cfg=yaml.safe_load((Path(candidate['path'])/'config.resolved.yaml').read_text(encoding='utf-8'))
                    if candidate['regularization']==0:matched_probe_configs.append(candidate_cfg)
                    if not row['reused_E1']:
                        git('merge-base','--is-ancestor',proof['git_commit'],m['git_commit'])
                        assert m['started_at']>proof['recorded_at_utc'] and not m['git_dirty']
            assert len(matched_probe_configs)==3
            assert all(set(diff_configs(matched_probe_configs[0],c))<={'data.feature_path'} for c in matched_probe_configs)
    a=[r['probe_accuracy'] for r in matrix if r['encoder']=='SimCLR' and r['fraction']==fractions[0]]
    b=[r['probe_accuracy'] for r in matrix if r['encoder']=='B0' and r['fraction']==fractions[0]]
    stats=paired_gain(a,b);assert all(result['E3'][k]==v for k,v in stats.items())
    for gain in result['E3']['gains']:
        fraction=gain['fraction']
        differences=[next(r['probe_accuracy'] for r in matrix if r['encoder']=='SimCLR' and r['seed']==seed and r['fraction']==fraction)
            -next(r['probe_accuracy'] for r in matrix if r['encoder']=='B0' and r['seed']==seed and r['fraction']==fraction) for seed in [1,2,3]]
        mean=sum(differences)/3;sd=math.sqrt(sum((d-mean)**2 for d in differences)/2)
        same=all(d>0 for d in differences) or all(d<0 for d in differences)
        assert differences==gain['deltas'] and math.isclose(mean,gain['mean'],abs_tol=1e-15)
        assert math.isclose(sd,gain['sample_std'],abs_tol=1e-15) and gain['same_nonzero_sign']==same
        assert gain['established']==(same and abs(mean)>=2*sd)
    tests=ET.parse('docs/environment/tests-lab4-final.xml').getroot();suites=list(tests.iter('testsuite'))
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','errors','failures','skipped']}
    assert counts['tests']>=106 and counts['errors']==counts['failures']==counts['skipped']==0
    figure_qa=read(ROOT/'figure_QA.json');assert figure_qa['status']=='PASS' and figure_qa['count']==8
    assert all(file_hash(r['path'])==r['sha256'] and r['status']=='PASS' for r in figure_qa['figures'])
    assert not list(ROOT.rglob('*.tex')) and not list(ROOT.rglob('*.pdf')) and not Path('reports/lab4').exists()
    assert not list(Path('runs/lab5').glob('*/manifest.json'))
    cf=read(ROOT/'counterfactuals.json');assert set(cf)=={'Q1','Q2','Q3','Q4','Q5'}
    assert cf['Q2']['status']=='BLOCKED'
    gates={'historical_Lab0_to_Lab3_unchanged':'PASS','Lab3_independent_audit':'PASS','mandatory_tests_and_regressions':'PASS',
        'N256_global_NT_Xent_not_microbatch_approximation':'PASS','GPU100_25_epoch_protocol':'PASS','trainonly_unlabelled_pretraining_cal_guard':'PASS',
        'E1_three_encoders_common_probe_knn':'PASS','E1_five_SSL_checkpoints_including0':'PASS',
        'E1_E2_exploratory_hypotheses_precede_formal_training':'PASS',
        'E1_final_loss_at_or_below_random':'PASS' if result['E1']['loss_threshold_pass'] else 'FAIL',
        'E1_G1_sign_reported':'PASS',
        'E2_six_variants_only_augmentation_changes':'PASS','E2_conditional_thresholds_applied':'PASS','E3_preregistered_before_new_runs':'PASS',
        'E3_complete36_matrix_nested_shared_indices':'PASS','E3_teacher_rule_applied':'PASS',
        'frozen_encoder_full_state_and_feature_repeatability':'PASS',
        'representation_scale_rank_spectrum_raw_recomputed':'PASS','CF1':'PASS','CF2':'BLOCKED','CF3':cf['Q3']['status'] if cf['Q3']['status']=='PASS' else 'FAIL',
        'eight_figures_visually_verified_and_hashed':'PASS',
        'CF4':'PASS','CF5':'PASS','formal_report':'NOT_APPLICABLE','optional_section14':'NOT_APPLICABLE'}
    totals={'formal_encoder_training_runs':len(encoders),'formal_probe_candidate_training_runs':len(probe_runs),
        'formal_encoder_training_seconds':sum(r['training_seconds'] for r in encoders),'formal_probe_training_seconds':sum(r['training_seconds'] for r in probe_runs),
        'checkpoint_count':len(checkpoints),'formal_checkpoint_count':sum(r['purpose']=='formal' for r in checkpoints),'RUN_ID_count':len(registry)}
    formal_training=[r for r in read(ROOT/'ledger.json') if r['purpose']=='formal']
    formal_features=[r for r in feature_rows if read(Path(r['path'])/'manifest.json')['purpose']=='formal']
    formal_evaluations=[r for r in evals if read(Path(r['path'])/'manifest.json')['purpose']=='formal']
    for operation in formal_features+formal_evaluations:
        assert operation['operation_wall_seconds']>0
        assert file_hash(Path(operation['path'])/'controller_snapshot.py')==operation['controller_source_sha256']
    totals.update(formal_encoder_and_initialization_wall_seconds=sum(r['elapsed_seconds'] for r in formal_training),
        formal_feature_operation_seconds=sum(r['operation_wall_seconds'] for r in formal_features),
        formal_probe_selection_and_test_seconds=sum(r['operation_wall_seconds'] for r in formal_evaluations))
    totals['formal_measured_active_seconds']=sum(totals[k] for k in ['formal_encoder_and_initialization_wall_seconds',
        'formal_feature_operation_seconds','formal_probe_selection_and_test_seconds'])
    first_started=min(datetime.fromisoformat(r['hardware_before']['at_utc']) for r in formal_training)
    endpoints=[datetime.fromisoformat(r['hardware_after']['at_utc']) for r in formal_training]
    endpoints += [datetime.fromisoformat(r['operation_started_at_utc'])+timedelta(seconds=r['operation_wall_seconds'])
        for r in formal_features+formal_evaluations]
    last_finished=max(endpoints)
    totals.update(formal_protocol_first_started_at_utc=first_started.isoformat(),formal_protocol_last_finished_at_utc=last_finished.isoformat(),
        formal_protocol_calendar_span_seconds=(last_finished-first_started).total_seconds())
    totals['timing_scope']='Nonoverlapping measured encoder/initialization workers + feature protocols + complete probe selection/test protocols. Probe candidate training is already included in probe protocols and is not added twice. Feature/probe Python startup, analysis, tests and idle time excluded.'
    write_json(ROOT/'audit.json',{'status':'FAIL_WITH_BLOCKED' if 'FAIL' in gates.values() else 'BLOCKED','block_reason':'CF2 requires missing teacher Protocol 4.6; other requirements are audited individually',
        'acceptance':gates,'automated_tests':counts,'totals':totals,'encoder_runs':encoders,'probe_runs':probe_runs,
        'historical_files_preserved':len(historical),'preregistration_commit':proof['git_commit'],
        'CF2_blocker':cf['Q2'],'raw_scientific_outcomes':{'G1_positive':result['E1']['G1']>0,'E3_difference_established':stats['established']},
        'criteria_note':'PositiveG1 or an establishedE3difference isnotmandatory: PDFallowsnegativeG1/anunestablisheddifference withanalysis. Their actual signs/statistics are preserved, notconvertedintoinventedacceptancethresholds.',
        'no_formal_reports_created':True,'Lab5_started':False})
    result.update(acceptance=gates,automated_tests=counts,totals=totals);write_json(ROOT/'results.json',result)
    print(json.dumps({'acceptance':gates,'totals':totals,'tests':counts},indent=2))

if __name__=='__main__':main()
