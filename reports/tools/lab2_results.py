"""Derive Lab 2 tables, scientific gates and figures exclusively from saved runs.

No LaTeX/PDF/report generation. Original Lab0/1 artifacts are read-only.
"""
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path
import torch
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analysis.learning_curves import first_crossover,paired_architecture,shift_metrics,leader_reversal
from src.utils.io import digest,file_hash,write_json

ROOT=Path('experiments/lab2');FIG=ROOT/'figures'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def csv_table(name,rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def git(*args):return subprocess.run(['git','-c','safe.directory=D:/神经网络',*args],capture_output=True,text=True,encoding='utf-8',check=True).stdout.strip()
def main():
    torch.set_num_threads(2);FIG.mkdir(exist_ok=True)
    ledger=read(ROOT/'ledger.json');runs={r['label']:r for r in ledger}
    evaluation=read(ROOT/'evaluations.json');ev={r['label']:r for r in evaluation}
    cfg=yaml.safe_load(Path('configs/lab2.yaml').read_text(encoding='utf-8'));fractions=cfg['profile']['lab2_fractions'];corruptions=cfg['profile']['lab2_corruptions']
    def acc(label):return ev[label]['accuracy']
    def summary(label):return read(Path(runs[label]['path'])/'summary.json')
    def resource(label):return read(Path(runs[label]['path'])/'resources.json')
    e1=[];delta=[]
    for r in fractions:
        pair={}
        for arm in ['cnn','vit']:
            label=f'E1_{arm}_r{r:g}_j1';s=summary(label);m=read(Path(runs[label]['path'])/'manifest.json')
            pair[arm]=acc(label+'_clean')
            e1.append({'model':arm,'fraction':r,'n_train':m['actual_train_size'],'seed':1,'RUN_ID':runs[label]['RUN_ID'],
                       'evaluation_RUN_ID':ev[label+'_clean']['RUN_ID'],'validation_accuracy':s['validation_metric'],
                       'test_accuracy':pair[arm],'optimizer_steps':s['global_step'],'data_epochs':s['data_epochs'],
                       'equivalent_epochs_formula':10000/math.ceil(m['actual_train_size']/128),'training_seconds':s['training_seconds'],
                       'inherited_optimizer_steps':runs[label].get('inherited_optimizer_steps',0),
                       'timing_scope':'child process only; parent interrupted timing separately disclosed' if runs[label].get('resumed_from') else 'complete training process'})
        delta.append(pair['vit']-pair['cnn'])
    csv_table('E1.csv',e1)
    selected_lr=read(ROOT/'selected_lr.json');exports={}
    for arm in ['cnn','vit']:
        parent=Path(selected_lr[arm]['path']);manifest=read(parent/'manifest.json');config=yaml.safe_load((parent/'config.resolved.yaml').read_text())
        exports[arm]={'RUN_ID':parent.name,'checkpoint':(parent/'checkpoints/last.pt').as_posix(),
            'checkpoint_sha256':file_hash(parent/'checkpoints/last.pt'),'resolved_config':(parent/'config.resolved.yaml').as_posix(),
            'model':config['model'],'feature_dimension':512 if arm=='cnn' else 384,
            'normalization':manifest['normalization'],'split_fingerprint':manifest['split_fingerprint'],
            'training_fraction':1.,'seed':1,'optimizer_steps':10000,'selection':'final validation LR grid; reused E1 r1',
            'source_hash':manifest['source_hash'],'git_commit':manifest['git_commit'],
            'corruption_protocol':'experiments/lab2/corruption_protocol.json','intended_future_use':'Lab8 robustness/calibration and Lab9 visual features; no later lab executed here'}
    write_json(ROOT/'exported_models.json',exports)
    e2=[];e2means=[]
    clean_delta=acc('E1_vit_r1_j1_clean')-acc('E1_cnn_r1_j1_clean')
    for c in corruptions:
        means={}
        for arm in ['cnn','vit']:
            label=f'E1_{arm}_r1_j1';clean=acc(label+'_clean');levels=[acc(f'{label}_{c}_{s}') for s in [1,2,3]]
            sm=shift_metrics(clean,levels);means[arm]=sm['mean_corrupted_accuracy']
            for s in [0,1,2,3]:
                ee=ev[f'{label}_{c}_{s}']
                e2.append({'model':arm,'corruption':c,'level':s,'accuracy':ee['accuracy'],'relative_drop':(clean-ee['accuracy'])/(clean+1e-12),
                           'mean_corrupted_accuracy':means[arm],'source_RUN_ID':ee['parent_RUN_ID'],'evaluation_RUN_ID':ee['RUN_ID']})
        d=means['vit']-means['cnn'];e2means.append({'corruption':c,'cnn_mean':means['cnn'],'vit_mean':means['vit'],'delta':d,'leader_reversal':leader_reversal(clean_delta,d)})
    csv_table('E2.csv',e2);write_json(ROOT/'E2_means.json',e2means)
    selection=read(ROOT/'selection.json');r=selection['fraction'];e3=[];values={'cnn':[],'vit':[]}
    for j in [1,2,3]:
        for arm in ['cnn','vit']:
            label=f'E3_{arm}_j{j}';el=f'E1_{arm}_r{r:g}_j1_clean' if j==1 else label+'_clean'
            values[arm].append(acc(el));e3.append({'model':arm,'seed':j,'fraction':r,'test_accuracy':acc(el),'RUN_ID':runs[label]['RUN_ID'],'evaluation_RUN_ID':ev[el]['RUN_ID'],'reused_E1':j==1})
    stats=paired_architecture(values['cnn'],values['vit']);csv_table('E3.csv',e3)
    stats.update(preregistered_prediction=selection['validation_delta'],
        hypothesis_supported=stats['established'] and stats['mean']*selection['validation_delta']>0,
        observed_minus_predicted=stats['mean']-selection['validation_delta'])
    assert math.isclose(stats['mean'],sum(stats['deltas'])/3,abs_tol=1e-15)
    assert math.isclose(stats['sample_std'],math.sqrt(sum((d-stats['mean'])**2 for d in stats['deltas'])/2),abs_tol=1e-15)
    assert selection['fraction']==max(selection['all_validation_candidates'],key=lambda v:(abs(v['validation_delta']),-v['fraction']))['fraction']
    paired_subsets=[]
    for j in [1,2,3]:
        pair=[read(Path(runs[f'E3_{arm}_j{j}']['path'])/'manifest.json') for arm in ['cnn','vit']]
        assert pair[0]['subset_fingerprint']==pair[1]['subset_fingerprint']
        assert pair[0]['seed']==pair[1]['seed']==j
        assert pair[0]['first_batch_checksum']==pair[1]['first_batch_checksum']
        paired_subsets.append(pair[0]['subset_fingerprint'])
    assert len(set(paired_subsets))==1
    matrix=[];conditions=[('clean',0)]+[(c,s) for c in corruptions for s in [1,2,3]]
    for r in fractions:
        for c,s in conditions:
            suffix='clean' if c=='clean' else f'{c}_{s}'
            cnn=ev[f'E1_cnn_r{r:g}_j1_{suffix}'];vit=ev[f'E1_vit_r{r:g}_j1_{suffix}']
            assert cnn['input_sha256']==vit['input_sha256']
            matrix.append({'fraction':r,'corruption':c,'level':s,'cnn':cnn['accuracy'],'vit':vit['accuracy'],'delta':vit['accuracy']-cnn['accuracy'],
                           'cnn_evaluation_RUN_ID':cnn['RUN_ID'],'vit_evaluation_RUN_ID':vit['RUN_ID'],'input_sha256':cnn['input_sha256']})
    csv_table('trainability_map.csv',matrix)
    resources={arm:resource(f'Resource_{arm}_isolated') for arm in ['cnn','vit']};p2=resource('CF4_ViT_patch2_profile')
    for arm in ['cnn','vit']:
        resources[arm].update(training_seconds=summary(f'E1_{arm}_r1_j1')['training_seconds'],
            training_peak_gpu_memory_bytes=summary(f'E1_{arm}_r1_j1')['peak_gpu_memory_bytes'],
            profiling_scope='Fresh process, initialized canonical model, no optimizer; training time/peak from actual r1 training RUN_ID')
        assert resources[arm]['MACs']==resource(f'E1_{arm}_r1_j1')['MACs']
    p_ratio=resources['vit']['parameters']/resources['cnn']['parameters'];mac_ratio=resources['vit']['MACs']/resources['cnn']['MACs']
    profiled_blocks={name:(v['MACs']-v['components']['conv']-384*10)/6 for name,v in [('P4',resources['vit']),('P2',p2)]}
    for name,v in [('P4',resources['vit']),('P2',p2)]:assert abs(profiled_blocks[name]/v['analytic']['block_MACs']-1)<=.01
    def observed_tokens(v):
        # Invert the independently hooked MHA MAC count per block, 4*n*d^2+2*n^2*d.
        d=384;a=v['components']['attention']/6
        return round((-4*d*d+math.sqrt(16*d**4+8*d*a))/(4*d))
    t_ratio=summary('E1_vit_r1_j1')['training_seconds']/summary('E1_cnn_r1_j1')['training_seconds']
    full_epochs=10000/math.ceil(43000/128);small=fractions[0];cf2=ev['CF2_equal_epochs_cnn_clean']
    translation=[]
    for arm in ['cnn','vit']:
        clean=acc(f'E1_{arm}_r1_j1_clean')
        for s,pixels in enumerate([2,4,8],1):
            ee=ev[f'E1_{arm}_r1_j1_translation_{s}'];translation.append({'model':arm,'pixels':pixels,'accuracy':ee['accuracy'],'absolute_drop':clean-ee['accuracy'],
                'relative_drop':(clean-ee['accuracy'])/(clean+1e-12),'multiple_of_patch4':pixels%4==0,'multiple_of_total_stride8':pixels%8==0,'covered_by_A1':pixels<=4,'RUN_ID':ee['parent_RUN_ID'],'evaluation_RUN_ID':ee['RUN_ID']})
    csv_table('CF5_translations.csv',translation)
    by_shift={p:statistics.mean(v['absolute_drop'] for v in translation if v['pixels']==p) for p in [2,4,8]}
    shift_observation=('The measured mean drop at8px exceeds both2px and4px. This is consistent with A1 covering2/4 but not8; divisibility by total CNN stride alone does not protect against the larger shift.'
        if by_shift[8]>max(by_shift[2],by_shift[4]) else
        'The measured8px mean drop is not larger than both smaller shifts. A1 coverage alone therefore does not order all effects; patch/stride alignment, boundaries and learned features also matter.')
    cf={'CF1':{'delta_small':delta[0],'delta_full':delta[-1],'difference':delta[-1]-delta[0],
           'source_RUN_IDs':[r['RUN_ID'] for r in e1 if r['fraction'] in [small,1.]],
           'interpretation':'No extrapolation beyond r1: CIFAR train is exhausted. Locality/weight sharing, A1, finite S, optimizer/LR, regularization, normalization and positional embeddings remain potential causes; architecture is not an isolated mechanism.'},
        'CF2':{'full_fraction_formula_epochs':full_epochs,'small_fraction_steps':summary('CF2_equal_epochs_cnn')['global_step'],
           'expected_steps':math.ceil(full_epochs*math.ceil(2150/128)),'equal_epoch_test_accuracy':cf2['accuracy'],'S10000_test_accuracy':acc('E1_cnn_r0.05_j1_clean'),
           'accuracy_change':cf2['accuracy']-acc('E1_cnn_r0.05_j1_clean'),'RUN_ID':cf2['parent_RUN_ID'],'evaluation_RUN_ID':cf2['RUN_ID'],
           'interpretation':'Shorter training changes repeated exposure and schedule horizon together: 26 warmup steps then cosine to506, versus500 then10000. Ceil rounding and final short batches explain fractional epoch discrepancy; this is not a pure mechanism estimate.'},
        'CF3':{'vit_to_cnn_parameter_ratio':p_ratio,'vit_to_cnn_MAC_ratio':mac_ratio,'vit_to_cnn_training_seconds_ratio':t_ratio,
           'param_matched':abs(p_ratio-1)<=.2,'compute_matched':abs(mac_ratio-1)<=.2,'resources':resources,
           'RUN_IDs':{arm:runs[f'E1_{arm}_r1_j1']['RUN_ID'] for arm in ['cnn','vit']}},
        'CF4':{'P4_tokens':resources['vit']['analytic']['tokens'],'P2_tokens':p2['analytic']['tokens'],
           'profile_inferred_tokens':{'P4':observed_tokens(resources['vit']),'P2':observed_tokens(p2)},
           'profiled_block_MACs':profiled_blocks,
           'P4_block_MACs':resources['vit']['analytic']['block_MACs'],'P2_block_MACs':p2['analytic']['block_MACs'],
           'quadratic_attention_ratio':p2['analytic']['attention_products_MACs']/resources['vit']['analytic']['attention_products_MACs'],
           'P2_resources':p2,'P2_RUN_ID':runs['CF4_ViT_patch2_profile']['RUN_ID'],
           'parameter_ratio':p2['parameters']/resources['vit']['parameters'],
           'interpretation':'Transformer block weight dimensions do not depend on token count. Only patch projection and learned positional table change, while attention products scale quadratically with token count.'},
        'CF5':{'translation_results':translation,'mean_absolute_drop_by_pixels':by_shift,'observed_comparison':shift_observation,'interpretation':'Both 2px and4px are covered by A1, 8px is not.4/8 are multiples of P4; only8 is a multiple of total CNN stride8. Zero padding, finite image boundaries, learned positions and pooling prevent a claim of exact classification invariance from equation2.2. Differences at8 cannot isolate augmentation from stride because both change together.'}}
    write_json(ROOT/'counterfactuals.json',cf)
    write_json(ROOT/'interpretation.json',{
        'E1':{'observed_deltas':delta,'r_star':first_crossover(fractions,delta),
              'comment':'These are sampled fractions at fixed S and A1, with differing repeated exposure. No extrapolation beyond the complete43000-image training role.'},
        'E2':{'clean_delta':clean_delta,'observed_means':e2means,
              'comment':'Patch shuffle preserves content within each tile but disrupts CNN neighborhoods. At4px it permutes intact ViT patches; self-attention without positions is permutation-equivariant, while learned absolute positions prevent exact invariance here. Measured differences are consistent with different spatial biases but do not isolate a single causal mechanism. All E2 leader changes are exploratory seed1 observations.'},
        'E3':{'selected_fraction':selection['fraction'],'paired_statistics':stats,
              'comment':'The three-seed conclusion concerns clean test at the validation-locked fraction. It does not confirm the single-seed E2 corruption leader changes.'},
        'resources':{'parameter_ratio':p_ratio,'MAC_ratio':mac_ratio,'time_ratio':t_ratio,
              'comment':'Equal parameter count within tolerance does not imply equal compute, timing or optimization behavior. No general architecture ranking outside this recipe is claimed.'}})
    # Verify complete training, checkpoint portability and immutable historical evidence.
    formal=[];checkpoint_index=[]
    for rr in ledger:
        if rr['kind']!='training':continue
        path=Path(rr['path']);m=read(path/'manifest.json')
        if m['purpose']!='formal':continue
        s=read(path/'summary.json');config=yaml.safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8'))
        assert digest(config)==m['config_hash'] and m['RUN_ID']==path.name
        target=config['train']['max_steps'];lines=[json.loads(x) for x in (path/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(lines)==target and lines[-1]['global_step']==target and all(x['global_step']==i+1 for i,x in enumerate(lines))
        assert m['status']=='COMPLETED' and not m['git_dirty'] and s['global_step']==target
        st=torch.load(path/'checkpoints/last.pt',map_location='cpu',weights_only=False)
        assert st['global_step']==target and st['source_hash']==m['source_hash'] and st['config_hash']==m['config_hash']
        assert all(torch.isfinite(v).all() for v in st['model'].values() if v.is_floating_point())
        if config['data']['train_fraction']==1.:
            from src.models.small_cnn import build_model
            model=build_model(config['model']).eval();model.load_state_dict(st['model'])
            with torch.no_grad():
                x=torch.zeros(2,3,32,32);features=model.forward_features(x)
                assert features.shape==(2,512 if config['model']['name']=='cifar_resnet18' else 384)
                assert torch.isfinite(features).all() and model(x).shape==(2,10)
            del model
        for kind in ['last','best']:
            checkpoint=path/f'checkpoints/{kind}.pt';assert checkpoint.exists()
            checkpoint_index.append({'label':rr['label'],'RUN_ID':path.name,'path':checkpoint.as_posix(),'sha256':file_hash(checkpoint),'model':config['model'],'fraction':config['data']['train_fraction'],'seed':config['experiment']['seed'],'steps':target,'inference_api':'build_model(config); load_checkpoint(path,model); model.forward_features(x)','full_fraction_for_Lab8_9':config['data']['train_fraction']==1.})
            checkpoint_index[-1]['selected_E1_r1']=path.name in {v['RUN_ID'] for v in selected_lr.values()}
        formal.append({'label':rr['label'],'RUN_ID':path.name,'steps':target,'training_seconds':s['training_seconds'],'wall_seconds':rr['elapsed_seconds']})
        del st
    historical=read('experiments/lab1_verification/historical_files.json')
    preserved=all(Path(p).is_file() and file_hash(p)==h for p,h in historical.items())
    clean_identity=all(acc(f'E1_{arm}_r1_j1_{c}_0')==acc(f'E1_{arm}_r1_j1_clean') and ev[f'E1_{arm}_r1_j1_{c}_0']['input_sha256']==ev[f'E1_{arm}_r1_j1_clean']['input_sha256'] for arm in ['cnn','vit'] for c in corruptions)
    from collections import defaultdict
    hashes=defaultdict(set)
    for ee in evaluation:
        assert ee['count']==10000
        hashes[(ee['corruption'],ee['level'])].add(ee['input_sha256'])
    assert all(len(v)==1 for v in hashes.values())
    prereg=read(ROOT/'preregistration_evidence.json')
    for rr in formal:
        m=read(Path(runs[rr['label']]['path'])/'manifest.json')
        git('merge-base','--is-ancestor',prereg['initial_conditional_commit'],m['git_commit'])
    for j in [2,3]:
        for arm in ['cnn','vit']:
            m=read(Path(runs[f'E3_{arm}_j{j}']['path'])/'manifest.json');git('merge-base','--is-ancestor',prereg['git_commit'],m['git_commit'])
            assert m['started_at']>prereg['recorded_at_utc']
    original=git('show',prereg['git_commit']+':experiments/lab2/hypothesis.md')
    assert original==Path('experiments/lab2/hypothesis.md').read_text(encoding='utf-8').strip()
    gates={'E1_parameter_match':abs(p_ratio-1)<=.2,'E1_MAC_analytic_error':all(v['MAC_relative_error']<=.01 for v in resources.values()),
        'E1_all_fractions_val_test':len(e1)==10,'E1_exact_S_all_primary':len(formal)==19 and sum(r['steps'] for r in formal)==180506,
        'E2_s0_exact_identity':clean_identity,'E2_all_metrics':len(e2)==24,'E3_preregistration_verified':True,
        'E3_statistical_difference_established':stats['established'],'map_all_fractions_and_conditions':len(matrix)==50,
        'E3_preregistered_hypothesis_supported':stats['hypothesis_supported'],
        'CF1_to_CF5_complete':True,'all_six_corruptions_table':len(list(ROOT.glob('all_corruptions_*.csv')))==2,
        'identical_paired_test_inputs_all_conditions':True,'isolated_resource_passports':True,
        'checkpoint_integrity_and_future_features':True,'historical_Lab0_Lab1_preserved':preserved}
    acceptance={k:('PASS' if v else 'FAIL') for k,v in gates.items()}
    acceptance.update({'compute_matched':'NOT_APPLICABLE_PRIMARY_PARAM_MATCHED',
       'leader_reversal':'OBSERVED' if any(r['leader_reversal'] for r in e2means) else 'NOT_OBSERVED_NOT_REQUIRED',
       'optional_face_transfer':'NOT_REQUESTED_TEACHER_PACK_UNAVAILABLE','report':'DEFERRED_BY_USER'})
    import xml.etree.ElementTree as ET
    xml=ET.parse('docs/environment/tests-lab2-final.xml').getroot()
    suites=list(xml.iter('testsuite'));test_counts={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
    acceptance['mandatory_and_regression_tests']='PASS' if test_counts['tests']>=70 and test_counts['failures']==test_counts['errors']==test_counts['skipped']==0 else 'FAIL'
    result={'E1':e1,'deltas':delta,'r_star':first_crossover(fractions,delta),'E2':e2means,'E3':stats,'E3_rows':e3,
       'selected_fraction':selection['fraction'],'formal_training_executions':len(formal),'formal_optimizer_updates':sum(r['steps'] for r in formal),
       'total_training_seconds':sum(r['training_seconds'] for r in formal),'training_execution_wall_seconds':sum(r['wall_seconds'] for r in formal),
       'evaluation_seconds':sum(r['wall_seconds'] for r in evaluation),'evaluation_executions':len(evaluation),
       'acceptance':acceptance,'formal_runs':formal,'historical_files_preserved':len(historical) if preserved else False,
       'automated_tests':test_counts,
       'fixed_variables':['canonical architectures and parameter-matched mode','fixed classwise nested subsets and split','S10000 B128 A1 AdamW wd.05 warmup500 cosine float32','architecture-specific LR selected once on validation','seed1 for map','same per-image corrupted inputs'],
       'map_varied_variables':['fraction','corruption identity','severity'],
       'limitations':['Three paired seeds describe initial weights and batch order, not alternative subsets or training recipes.','E2 leader changes use exploratory seed1 only; clean-regime E3 does not confirm corruption-specific rankings.','Single CIFAR split and synthetic label-preserving corruptions do not establish natural-domain transfer.','Unequal MACs/time remain in the parameter-matched comparison.','Seed1 reused from exploratory E1 as required; initial conditional rule precedes E1, finalized hypothesis precedes new seeds2/3.']}
    if (ROOT/'interruption.json').exists():
        interrupted=read(ROOT/'interruption.json');result['interruption']=interrupted
        old=Path('runs/lab2')/interrupted['source_RUN_ID'];child=Path('runs/lab2')/interrupted['recovered_RUN_ID']
        original_steps=(old/'steps.jsonl').read_text().splitlines();resumed_steps=(child/'steps.jsonl').read_text().splitlines()
        assert original_steps==resumed_steps[:len(original_steps)]
        interrupted.update(overlapping_step_records_verified=len(original_steps),overlapping_trajectory_exact=True)
        write_json(ROOT/'interruption.json',interrupted)
        result['formal_process_invocations_including_interrupted']=len(formal)+1
        result['recorded_physical_optimizer_updates_including_replay']=sum(v['steps'] for v in formal)+interrupted['recorded_updates_replayed']
        result['timing_note']='total_training_seconds sums completed process training durations; recovered child excludes inherited prefix compute. Interrupted parent exact monotonic duration was lost. Its complete-data-epoch seconds are reported separately as a lower bound; never infer a fabricated precise all-process total.'
        result['training_seconds_lower_bound_including_interrupted_epochs']=result['total_training_seconds']+interrupted['original_complete_data_epoch_seconds']
        result['limitations'].append('One external interruption required checkpoint-boundary replay of499 recorded updates; exact interrupted-process total time is unavailable. Scientific trajectory resumes full state; timing includes a separately disclosed incomplete parent duration.')
    write_json(ROOT/'checkpoint_index.json',checkpoint_index);write_json(ROOT/'results.json',result)
    # Publication-independent scientific plots, all derived from the tables above.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for arm in ['cnn','vit']:
        rows=[v for v in e1 if v['model']==arm];axes[0].plot(fractions,[v['test_accuracy'] for v in rows],'o-',label=arm.upper())
    axes[0].set(xscale='log',xlabel='Training fraction r',ylabel='Test accuracy');axes[0].legend()
    axes[1].plot(fractions,delta,'o-',color='#7553A3');axes[1].axhline(0,color='grey',lw=.7);axes[1].set(xscale='log',xlabel='Training fraction r',ylabel='ViT - CNN test accuracy')
    for ax in axes:
        ax.set_xticks(fractions,[f'{v:g}' for v in fractions]);ax.minorticks_off()
    fig.tight_layout();fig.savefig(FIG/'E1_learning_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,c in zip(axes,corruptions):
        for arm in ['cnn','vit']:ax.plot(range(4),[v['accuracy'] for v in e2 if v['model']==arm and v['corruption']==c],'o-',label=arm.upper())
        ax.set(title=c,xlabel='Severity',ylabel='Test accuracy',xticks=range(4));ax.legend()
    fig.tight_layout();fig.savefig(FIG/'E2_corruptions.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.5));ax.bar([1,2,3],stats['deltas'],color='#7553A3');ax.axhline(0,color='grey');ax.set(xlabel='Paired seed',ylabel='ViT - CNN test accuracy',xticks=[1,2,3]);fig.tight_layout();fig.savefig(FIG/'E3_paired_seeds.png');plt.close(fig)
    grid=np.array([v['delta'] for v in matrix]).reshape(5,10);lim=max(abs(grid.min()),abs(grid.max()))
    fig,ax=plt.subplots(figsize=(11,4.3));im=ax.imshow(grid,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    ax.set(xticks=range(10),xticklabels=['clean']+[f'{c}:{s}' for c,s in conditions[1:]],yticks=range(5),yticklabels=fractions,ylabel='Training fraction r')
    ax.set_title('Seed 1; 10,000 updates; B128; A1; fixed architectures and learning rates',fontsize=10)
    plt.setp(ax.get_xticklabels(),rotation=40,ha='right');fig.colorbar(im,ax=ax,label='ViT - CNN accuracy');fig.tight_layout();fig.savefig(FIG/'trainability_map.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.5))
    for arm in ['cnn','vit']:
        s=summary(f'E1_{arm}_r1_j1');ax.scatter(s['training_seconds']/60,acc(f'E1_{arm}_r1_j1_clean'),label=arm.upper(),s=65)
    ax.set(xlabel='Training time (minutes)',ylabel='Test accuracy at r1');ax.legend();fig.tight_layout();fig.savefig(FIG/'time_accuracy.png');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for i,arm in enumerate(['cnn','vit']):
        for r in fractions:
            p=Path(runs[f'E1_{arm}_r{r:g}_j1']['path'])/'steps.jsonl';steps=[json.loads(v) for v in p.read_text().splitlines()]
            losses=np.array([v['train_loss'] for v in steps]);smoothed=losses[:len(losses)//100*100].reshape(-1,100).mean(1)
            axes[i].plot(np.arange(100,len(smoothed)*100+1,100),smoothed,label=f'r={r:g}')
        axes[i].set(ylabel=f'{arm.upper()} train loss');axes[i].legend(ncol=5,fontsize=8)
    axes[-1].set_xlabel('Optimizer update (100-update mean loss)');fig.tight_layout();fig.savefig(FIG/'training_loss.png');plt.close(fig)
    print(json.dumps({'E3':stats,'formal_training_executions':len(formal),'acceptance':acceptance},indent=2))

if __name__=='__main__':main()
