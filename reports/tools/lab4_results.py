"""Derive Lab4 tables, figures and counterfactual answers from actual saved RUNs."""
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from src.analysis.label_efficiency import paired_gain,classify_ablation
from src.compare import diff_configs,compare_runs
from src.utils.io import write_json

ROOT=Path('experiments/lab4');FIG=ROOT/'figures'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load_rows(name):return read(ROOT/name) if (ROOT/name).exists() else []
def csv_table(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def diagnostic(path,epoch):return read(Path(path)/f'diagnostics/epoch_{epoch:03}.json')['representation']
def collapse(rep):
    h,z=rep['h'],rep['z_hat']
    return bool(h['collapse_scale'] or z['collapse_scale'] or z['mean_vector_norm']>=.9 or
                any(v['effective_rank_ratio'] is not None and v['effective_rank_ratio']<.2 for v in [h,z]))

def main():
    FIG.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    ledger={r['label']:r for r in load_rows('ledger.json') if r['status'].startswith('COMPLETED')}
    evaluations={r['label']:r for r in load_rows('evaluations.json') if r['status'].startswith('COMPLETED')}
    summary={'report':'DEFERRED_BY_USER','Q2':'BLOCKED missingteacherProtocol§4.6'}
    all_encoders=[]
    if (ROOT/'E1_sources.json').exists():
        sources=read(ROOT/'E1_sources.json');e1=[]
        for name,label,epoch in [('B0','E1_B0_h',0),('B1','E1_B1_h',25),('SimCLR','E1_SSL_epoch100_h',100)]:
            ev=evaluations[label];rep=diagnostic(sources[name],epoch)
            row={'encoder':name,'encoder_RUN_ID':Path(sources[name]).name,'evaluation_RUN_ID':ev['RUN_ID'],'epoch':epoch,
                'probe_accuracy':ev['test_accuracy'],'knn_accuracy':ev['knn_test_accuracy'],'h_trace_cov':rep['h']['trace_cov'],
                'h_effective_rank':rep['h']['effective_rank'],'h_effective_rank_ratio':rep['h']['effective_rank_ratio'],
                'z_trace_cov':rep['z_hat']['trace_cov'],'z_effective_rank':rep['z_hat']['effective_rank'],
                'z_effective_rank_ratio':rep['z_hat']['effective_rank_ratio'],'mean_zhat_norm':rep['z_hat']['mean_vector_norm'],
                'collapse_scale_h':rep['h']['collapse_scale'],'collapse_scale_z':rep['z_hat']['collapse_scale'],'any_PDF_collapse_diagnostic':collapse(rep)}
            e1.append(row);all_encoders.append(row)
            folder=ROOT/'encoders'/name;folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/'summary.json',{'encoder_RUN_ID':row['encoder_RUN_ID'],'evaluation_RUN_ID':ev['RUN_ID'],
                'projector_status':'contrastively trained' if name=='SimCLR' else 'random untrained projector; baseline quality is evaluated on h',
                'representation':{'h':{**rep['h'],'linear_probe':ev['test_accuracy'],'knn':ev['knn_test_accuracy']},'z_hat':rep['z_hat']},
                'feature_cache_RUN_ID':ev['feature_RUN_ID'],'checkpoint':ev['checkpoint'],'probe_protocol':'commontrain-onlystandardizedfrozenfeatures;val-selectedL2;testfinal'})
        csv_table('E1.csv',e1)
        ssl=Path(sources['SimCLR']);metrics=[json.loads(v) for v in (ssl/'metrics.jsonl').read_text().splitlines()]
        curve=[]
        for epoch in [0,25,50,75,100]:
            ev=evaluations[f'E1_SSL_epoch{epoch}_h'];rep=diagnostic(ssl,epoch)
            curve.append({'epoch':epoch,'probe_accuracy':ev['test_accuracy'],'validation_accuracy':ev['validation_accuracy'],'evaluation_RUN_ID':ev['RUN_ID'],
                'h_effective_rank':rep['h']['effective_rank'],'z_effective_rank':rep['z_hat']['effective_rank']})
            if epoch not in [0,100]:all_encoders.append({'encoder':f'SimCLR_epoch{epoch}','encoder_RUN_ID':ssl.name,'evaluation_RUN_ID':ev['RUN_ID'],
                    'probe_accuracy':ev['test_accuracy'],'h_effective_rank':rep['h']['effective_rank'],'h_trace_cov':rep['h']['trace_cov']})
        csv_table('E1_probe_checkpoints.csv',curve)
        loss={v['epoch']:v['train_loss'] for v in metrics};plateaus=[];intervals=[]
        for a,b in zip(curve[1:],curve[2:]):
            row={'epoch1':a['epoch'],'epoch2':b['epoch'],'loss_decrease':loss[a['epoch']]-loss[b['epoch']],
                'probe_change':b['probe_accuracy']-a['probe_accuracy']}
            intervals.append(row)
            if row['loss_decrease']>=.05 and abs(row['probe_change'])<=.01:plateaus.append(row)
        summary['E1']={'rows':e1,'G1':e1[2]['probe_accuracy']-e1[0]['probe_accuracy'],'final_NT_Xent_loss':metrics[-1]['train_loss'],
            'B1_minus_SSL_probe':e1[1]['probe_accuracy']-e1[2]['probe_accuracy'],
            'interpretation':'The full-label probe gain is descriptive for one fixed pretraining seed. B1 uses labels during encoder training; its gap from SSL is not a matched-label-budget cost estimate. Rank and collapse flags describe feature geometry and must be interpreted together with the measured downstream accuracies, not as a replacement for them.',
            'random_baseline_loss':math.log(511),'loss_threshold_pass':metrics[-1]['train_loss']<=math.log(511),
            'probe_checkpoints':curve,'plateau_analysis':{'definition':'lossdrop>=.05 andabsprobechange<=.01 between adjacentmeasuredcheckpoints25..100',
                'observed':bool(plateaus),'matching_intervals':plateaus,'all_intervals':intervals}}
        fig,axes=plt.subplots(1,2,figsize=(10,3.7));epochs=[r['epoch'] for r in metrics]
        axes[0].plot(epochs,[r['train_loss'] for r in metrics],label='SimCLR');axes[0].axhline(math.log(511),ls='--',color='grey',label='ln511 random baseline')
        axes[0].set(xlabel='Pretraining epoch',ylabel='NT-Xent loss');axes[0].legend(fontsize=8)
        axes[1].plot([r['epoch'] for r in curve],[r['probe_accuracy'] for r in curve],'o-');axes[1].set(xlabel='Pretraining checkpoint epoch',ylabel='Frozen h probe test accuracy')
        fig.tight_layout();fig.savefig(FIG/'E1_loss_probe.png');plt.close(fig)
        diagnostics=[read(ssl/f'diagnostics/epoch_{epoch:03}.json') for epoch in range(101)]
        fig,axes=plt.subplots(1,2,figsize=(10,3.7))
        axes[0].plot(epochs,[r['positive_similarity'] for r in metrics],label='training positive');axes[0].plot(epochs,[r['negative_similarity'] for r in metrics],label='training negative')
        for kind in ['h','z_hat']:
            axes[1].plot(range(101),[r['representation'][kind]['positive_similarity'] for r in diagnostics],label=kind+' fixed positive')
            axes[1].plot(range(101),[r['representation'][kind]['random_pair_similarity'] for r in diagnostics],'--',label=kind+' fixed random')
        for ax in axes:ax.set(xlabel='Epoch',ylabel='Mean cosine similarity');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(FIG/'E1_similarities.png');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,3.7))
        for kind in ['h','z_hat']:
            axes[0].plot(range(101),[r['representation'][kind]['effective_rank_ratio'] for r in diagnostics],label=kind)
            axes[1].plot(range(101),[r['representation'][kind]['trace_cov'] for r in diagnostics],label=kind)
        axes[0].axhline(.2,ls='--',color='grey',label='diagnostic threshold 0.2')
        axes[0].set(xlabel='Epoch',ylabel='Effective rank / dimension');axes[1].set(xlabel='Epoch',ylabel='Covariance trace',yscale='log')
        for ax in axes:ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(FIG/'E1_rank_trace.png');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,3.7))
        for epoch in [0,50,100]:
            rep=diagnostic(ssl,epoch)
            for ax,kind in zip(axes,['h','z_hat']):
                values=rep[kind]['eigenvalues'];ax.plot(range(1,len(values)+1),np.maximum(values,1e-18),label=f'epoch{epoch}')
                ax.set(xlabel='Component index',ylabel='Covariance eigenvalue',yscale='log',title=kind);ax.legend(fontsize=8)
        fig.suptitle('Fixed 5000 train images; display floor 1e-18 only; raw zeros retained',fontsize=10)
        fig.tight_layout();fig.savefig(FIG/'E1_spectra.png');plt.close(fig)
    # E2 canbe generated once allsixvariations andE1baseline exist.
    variants=[r for name,r in evaluations.items() if name.startswith('E2_') and name.endswith('_h')]
    if len(variants)==6 and 'E1' in summary:
        e2=[];full=evaluations['E2_T_h'];g0=full['test_accuracy']-evaluations['E1_B0_h']['test_accuracy'];diffs={}
        full_collapse=collapse(diagnostic(ledger['E2_T']['path'],25))
        full_cfg=yaml.safe_load((Path(ledger['E2_T']['path'])/'config.resolved.yaml').read_text(encoding='utf-8'))
        for ev in variants:
            name=ev['label'][3:-2];train=ledger['E2_'+name];rep=diagnostic(train['path'],25)
            cfg=yaml.safe_load((Path(train['path'])/'config.resolved.yaml').read_text(encoding='utf-8'));difference=diff_configs(full_cfg,cfg)
            assert set(difference)<={'augmentations'},difference
            diffs[name]=difference;is_collapse=collapse(rep);delta=full['test_accuracy']-ev['test_accuracy']
            row={'variant':name,'training_RUN_ID':train['RUN_ID'],'evaluation_RUN_ID':ev['RUN_ID'],
                'probe_accuracy':ev['test_accuracy'],'validation_probe_accuracy':ev['validation_accuracy'],'knn_accuracy':ev['knn_test_accuracy'],
                'delta_probe':delta,'delta_knn':full['knn_test_accuracy']-ev['knn_test_accuracy'],
                'h_effective_rank':rep['h']['effective_rank'],'h_trace_cov':rep['h']['trace_cov'],'h_rank_ratio':rep['h']['effective_rank_ratio'],
                'z_rank_ratio':rep['z_hat']['effective_rank_ratio'],'mean_zhat_norm':rep['z_hat']['mean_vector_norm'],
                'collapse':is_collapse,'full_T_collapse':full_collapse,'new_collapse_relative_to_full_T':is_collapse and not full_collapse,
                'positive_half_gain_threshold_crossed':g0>0 and delta>=g0/2,
                'negative_half_gain_threshold_crossed':g0>0 and delta<=-g0/2,
                'classification':classify_ablation(g0,delta,is_collapse) if name.startswith('minus_') else 'boundary_or_reference'}
            e2.append(row);all_encoders.append({'encoder':'E2_'+name,'encoder_RUN_ID':train['RUN_ID'],'evaluation_RUN_ID':ev['RUN_ID'],
                'probe_accuracy':ev['test_accuracy'],'h_effective_rank':rep['h']['effective_rank'],'h_trace_cov':rep['h']['trace_cov']})
            folder=ROOT/'encoders'/('E2_'+name);folder.mkdir(exist_ok=True)
            write_json(folder/'summary.json',{'encoder_RUN_ID':train['RUN_ID'],'representation':{'h':{**rep['h'],'linear_probe':ev['test_accuracy'],'knn':ev['knn_test_accuracy']},'z_hat':rep['z_hat']},'evaluation_RUN_ID':ev['RUN_ID']})
        csv_table('E2.csv',e2);write_json(ROOT/'E2_config_differences.json',diffs)
        harness_comparison=compare_runs([ledger['E2_'+r['variant']]['path'] for r in e2])
        assert all(set(d)<={'augmentations'} for d in harness_comparison['config_differences_from_first'])
        write_json(ROOT/'E2_diff_runs.json',harness_comparison)
        summary['E2']={'rows':e2,'G0':g0,'half_gain_rule_applicable':g0>0,'invariances':{'random_resized_crop':'translation,scale,cropcontent invariance','color_jitter':'brightness,contrast,saturation,hue invariance','flip':'horizontalmirror invariance'},
            'boundary_selection':read(ROOT/'E2_single_selection.json'),'limitations':'Singleseed1 exploratoryaugmentation effects; collapse threshold isdiagnostic, notproofthat usefulclassificationinformation isabsent. IfbothTandablation triggerdimensionalcollapse, theliteralteacherflagdoesnotestablishthat removalcausednewcollapse; full_T_collapse andnew_collapse_relative_to_full_T are reported separately.'}
        b0_accuracy=evaluations['E1_B0_h']['test_accuracy']
        summary['E2']['boundary_analysis']={
            'B0_probe_accuracy':b0_accuracy,
            'measurements':[{'variant':r['variant'],'probe_minus_B0':r['probe_accuracy']-b0_accuracy,
                'delta_from_T':r['delta_probe'],'collapse':r['collapse']} for r in e2
                if r['variant']=='none' or r['variant'].startswith('only_')],
            'equation_4_6':'With identical views, the positive equals the anchor. Its attraction is parallel to the normalized anchor and therefore vanishes in the tangent space after differentiating L2 normalization. The softmax-weighted negative terms still repel other images. Thus identical views can learn instance separation without useful augmentation invariance, and complete collapse is not inevitable. The measured scale, mean vector norm and rank flags distinguish these outcomes; a low rank alone does not prove absence of class information.'}
        loo=[r for r in e2 if r['variant'].startswith('minus_')];fig,ax=plt.subplots(figsize=(7,3.8));x=np.arange(3)
        ax.bar(x-.18,[r['delta_probe']*100 for r in loo],.36,label='probe');ax.bar(x+.18,[r['delta_knn']*100 for r in loo],.36,label='kNN')
        ax.axhline(0,color='grey',lw=.7);ax.set(xticks=x,xticklabels=[r['variant'][6:] for r in loo],ylabel='M(T) - M(T minus operation), percentage points');ax.legend()
        fig.tight_layout();fig.savefig(FIG/'E2_ablation.png');plt.close(fig)
        names={'T':'T','minus_random_resized_crop':'T - crop','minus_color_jitter':'T - color',
            'minus_flip':'T - flip','only_random_resized_crop':'only crop','none':'none'}
        fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(e2))
        ax.bar(x-.19,[r['probe_accuracy']*100 for r in e2],.38,label='h probe')
        ax.bar(x+.19,[r['knn_accuracy']*100 for r in e2],.38,label='h kNN')
        ax.axhline(evaluations['E1_B0_h']['test_accuracy']*100,color='grey',ls='--',label='B0 probe')
        ax.axhline(evaluations['E1_B0_h']['knn_test_accuracy']*100,color='grey',ls=':',label='B0 kNN')
        ax.set(xticks=x,xticklabels=[names.get(r['variant'],r['variant']) for r in e2],ylabel='Test accuracy (%)',
            title='E2: six augmentation conditions, 25 epochs, seed 1')
        ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(FIG/'E2_all_variants.png');plt.close(fig)
    if (ROOT/'preregistration_evidence.json').exists():
        fractions=read(ROOT/'profile.json')['lab4_label_fractions'];matrix=[]
        for seed in [1,2,3]:
            for fraction in fractions:
                for name in ['B0','B1','SimCLR']:
                    label=f'E3_{name}_s{seed}_r{fraction:g}'
                    if seed==1 and fraction==1.:
                        label='E1_SSL_epoch100_h' if name=='SimCLR' else f'E1_{name}_h'
                    if label not in evaluations:continue
                    ev=evaluations[label];matrix.append({'encoder':name,'seed':seed,'fraction':fraction,'probe_accuracy':ev['test_accuracy'],
                        'RUN_ID':ev['RUN_ID'],'encoder_RUN_ID':ev['encoder_RUN_ID'],'subset_fingerprint':ev['label_subset_fingerprint'],
                        'reused_E1':seed==1 and fraction==1.})
        if len(matrix)==36:
            csv_table('E3.csv',matrix);values=lambda name,r:[next(v['probe_accuracy'] for v in matrix if v['encoder']==name and v['seed']==s and v['fraction']==r) for s in [1,2,3]]
            stats=paired_gain(values('SimCLR',fractions[0]),values('B0',fractions[0]));hyp=read(ROOT/'E3_selection.json')
            gains=[]
            for r in fractions:
                paired=paired_gain(values('SimCLR',r),values('B0',r));gains.append({'fraction':r,**paired})
            stats.update(matrix=matrix,rmin=fractions[0],gains=gains,hypothesis_supported=stats['established'] and stats['mean']*hyp['predicted_sign']>0,
                preregistered_point_prediction=hyp['predicted_mean_difference'],
                point_prediction_error=stats['mean']-hyp['predicted_mean_difference'],
                ssl_better_established=stats['established'] and stats['mean']>0,
                conclusion='различие установлено' if stats['established'] else 'различие не установлено',
                limitations=['Seeds change selectedtraininglabels andprobeinitialization, notSSLpretraining; onlyonefixedencoder perarm.',
                    '5000labelledvalidationimages selectregularization; thisoptimisticfew-labelcondition issharedbyallarms.'])
            summary['E3']=stats;write_json(ROOT/'E3_statistics.json',stats);csv_table('E3_gain.csv',gains)
            fig,axes=plt.subplots(1,2,figsize=(10,4));markers=['o','s','^']
            for name in ['B0','B1','SimCLR']:
                line=axes[0].plot(fractions,[np.mean(values(name,r)) for r in fractions],label=name)[0]
                for seed,marker in zip([1,2,3],markers):
                    axes[0].scatter(fractions,[next(v['probe_accuracy'] for v in matrix if v['encoder']==name and v['seed']==seed and v['fraction']==r) for r in fractions],marker=marker,color=line.get_color(),s=30)
            axes[1].plot(fractions,[g['mean']*100 for g in gains],label='mean SimCLR - random')
            for s,marker in enumerate(markers):axes[1].scatter(fractions,[g['deltas'][s]*100 for g in gains],marker=marker,s=30)
            for ax in axes:ax.set_xscale('log');ax.set_xticks(fractions,[f'{r:.0%}' for r in fractions]);ax.minorticks_off();ax.set_xlabel('Labelled train fraction');ax.legend(fontsize=8)
            axes[0].set_ylabel('Probe test accuracy');axes[1].set_ylabel('G(r), percentage points');axes[1].axhline(0,color='grey',lw=.7)
            fig.suptitle('Fixed E1 encoders; o/s/^: subset + probe seeds 1/2/3; B1 pretrained with all labels',fontsize=10)
            fig.tight_layout();fig.savefig(FIG/'E3_label_efficiency.png');plt.close(fig)
    # Actualcounterfactual values; missingteacherdefinition never filledwith inventeddata.
    cf={'Q2':read(ROOT/'missing_resources.json')['Q2']}
    if 'E1_SSL' in ledger and 'CF1_N64' in ledger:
        comparisons=[]
        for label,n in [('E1_SSL',256),('E2_T',256),('CF1_N64',64)]:
            path=Path(ledger[label]['path']);steps=[json.loads(v) for v in (path/'steps.jsonl').read_text().splitlines()]
            comparisons.append({'RUN_ID':path.name,'N':n,'negatives':2*n-2,'random_baseline_loss':math.log(2*n-1),'first_loss':steps[0]['train_loss'],'last_loss':steps[-1]['train_loss'],
                'epoch_budget':100 if label=='E1_SSL' else 25,'first_step_skipped':steps[0]['optimizer_step_skipped'],
                'baseline_minus_last_loss':math.log(2*n-1)-steps[-1]['train_loss']})
        cf['Q1']={'status':'PASS','measurements':comparisons,'answer':'NT-Xent denominators contain2N-1 candidates, andtauchanges attainableloss andgradientscale. DifferentN/tau lossvalues are not directly comparable. E1full100first/lastloss isreported asrequired; E2T25 providesanexisting horizon-matchedN256referenceforCF1N64/25. Batchnormalizationgrouping andnumberofupdatesalsochange withN, sothis isnot anisolateddenominator-onlymechanismestimate.'}
    if len(all_encoders)>=2:
        candidates=[]
        for a,b in itertools.combinations(all_encoders,2):
            ra,rb=a['h_effective_rank'],b['h_effective_rank']
            if ra is None or rb is None:continue
            rankgap=abs(ra-rb)/max(ra,rb);accgap=abs(a['probe_accuracy']-b['probe_accuracy'])
            if rankgap<=.10 and accgap>=.02:candidates.append({'criterion':'rankrelativegap<=.10 andaccuracygap>=.02','rank_relative_gap':rankgap,'accuracy_gap':accgap,'A':a,'B':b})
            elif accgap<=.01 and rankgap>=.10:candidates.append({'criterion':'accuracygap<=.01 andrankrelativegap>=.10','rank_relative_gap':rankgap,'accuracy_gap':accgap,'A':a,'B':b})
        cf['Q3']={'status':'PASS' if candidates else 'FAIL','actual_pairs':candidates,
            'answer':'Effective rank measures covariance dispersion, notwhetherdirections encode labels. Itmustbe shownwithtrace andmeannorm. When trace<=1e-6*d*mean_squared_norm, rankisundefined andreportednull/collapse_scale, neveran arbitraryepsilon-regularizednumber.'}
    if 'CF4_SSL_z_hat' in evaluations and 'E1_SSL_epoch100_h' in evaluations:
        h,z=evaluations['E1_SSL_epoch100_h'],evaluations['CF4_SSL_z_hat']
        cf['Q4']={'status':'PASS','h_probe_accuracy':h['test_accuracy'],'z_hat_probe_accuracy':z['test_accuracy'],'difference_h_minus_z':h['test_accuracy']-z['test_accuracy'],
            'RUN_IDs':[h['RUN_ID'],z['RUN_ID']],'answer':'Theprojection head optimizesaugmentationinvariance andchangesdimension/geometry. Thisprobecomparison canreveal aclassificationinformation difference underthislinearprocedure, butalowerzscore alone doesnotshowtheheadharmscontrastivelearning oralltasks.'}
    if 'E3' in summary and 'CF5_B1_low_labels_h' in evaluations:
        low=evaluations['CF5_B1_low_labels_h'];gains=summary['E3']['gains'];rmin=gains[0]['fraction']
        cf['Q5']={'status':'PASS','G_rmin':gains[0],'G1':gains[-1],'low_label_supervised_control':{k:low[k] for k in ['test_accuracy','RUN_ID','encoder_RUN_ID','label_subset_fingerprint']},
            'encoder_budget_comparison':{label:{'RUN_ID':ledger[label]['RUN_ID'],
                'successful_optimizer_updates':ledger[label]['global_step'],
                'physical_batches':sum(1 for _ in (Path(ledger[label]['path'])/'steps.jsonl').open(encoding='utf-8'))}
                for label in ['E1_B1','CF5_B1_low_labels']},
            'comparison_seed1':{name:next(v for v in summary['E3']['matrix'] if v['encoder']==name and v['seed']==1 and v['fraction']==rmin) for name in ['B0','B1','SimCLR']},
            'conditions':'CF5 B1 encoder uses only S1(rmin) labels for 25 epochs, same B1 optimizer/architecture/cropflip/schedule. Probe uses exactly the same 860 indices. Compared to full B1, unique encoder labels change 43000 -> 860 and physical batches 4200 -> 100; successful optimizer updates are reported separately including AMP skips. Equal epochs do not match exposure or compute. SSL sees 43000 unlabelled images for 100 epochs. All probes still use 5000 val labels for L2 selection.',
            'answer':'OriginalB1features alreadycontain supervisionfromall43000trainlabels evenwhenprobehasfewlabels, sotheB1-SSLgap isnot apriceofrefusinglabels. Thelow-labelencodercontrol reduces thatlabel-budgetconfounding butdoesnot matchtotalcompute ordataexposure.'}
    write_json(ROOT/'counterfactuals.json',cf);summary['counterfactual_status']={k:v['status'] for k,v in cf.items()}
    write_json(ROOT/'results.json',summary)
    print(json.dumps({'available_sections':list(summary),'counterfactual_status':summary['counterfactual_status'],'figures':len(list(FIG.glob('*.png')))},indent=2))

if __name__=='__main__':main()
