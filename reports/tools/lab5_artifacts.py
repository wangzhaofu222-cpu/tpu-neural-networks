"""Generate tables, figures and evidence-based counterfactual answers from raw RUNs."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.io import write_json,file_hash

R=Path('experiments/lab5');F=R/'figures'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(x):return f'{100*x:.2f}%'
def table(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def save(fig,name,caption):
    fig.suptitle('Cora · transductive · PyG public masks (course split unverified)',fontsize=11)
    fig.tight_layout(rect=(0,.03,1,.95));fig.text(.5,.006,caption,ha='center',fontsize=8)
    fig.savefig(F/(name+'.svg'));fig.savefig(F/(name+'.png'),dpi=170);plt.close(fig)

def main():
    F.mkdir(exist_ok=True);e1=read(R/'E1.json');e2=read(R/'E2.json');e3=read(R/'E3.json');idx=read(R/'run_index.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
    baseline=[]
    for kind,r in e1['models'].items():
        training=read(Path('runs/lab5')/r['source_RUN_ID']/'summary.json')
        baseline.append({'model':kind,**{k:r[k] for k in ['train','val','test','best_epoch','RUN_ID','source_RUN_ID']},'Gtop':e1['Gtop'].get(kind),'training_seconds':training['training_seconds']})
    table('E1_models.csv',baseline);table('E1_deletion.csv',e1['deletion'])
    paired=[]
    for r in e2['paired']:
        paired.append({'seed':r['seed'],'original_test':r['original']['test'],'original_val':r['original']['val'],'null_test':r['null']['test'],
           'MLP_test':r['mlp']['test'],'delta':r['delta'],'hE':r['null_hE'],'hdeg':r['hdeg'],'swap_success':r['swap_success'],
           'original_RUN_ID':r['original']['RUN_ID'],'null_RUN_ID':r['null']['RUN_ID'],'MLP_RUN_ID':r['mlp']['RUN_ID']})
    table('E2_pairs.csv',paired)
    depths=[{k:r[k] for k in ['depth','train','val','test','best_epoch','rhoT','GI','last_training_rhoT','GI_training_median','S','energy','probe_val','diagnosis','RUN_ID','source_RUN_ID','probe_RUN_ID']} for r in e3['plain']]
    table('E3_depths.csv',depths)
    layers=[]
    for name,s in idx['smoothing'].items():
        for r in s['layers']:layers.append({'condition':name,'analysis_RUN_ID':s['RUN_ID'],'source_RUN_ID':s['source_RUN_ID'],**r})
    table('E3_all_layers.csv',layers)
    p=[r['p'] for r in e1['deletion']]
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for key,label,c,m in [('A_test','A: original weights on deleted graph',colors[0],'o'),('B_test','B: train and evaluate deleted graph',colors[1],'s')]:
        ax[0].plot(p,[100*r[key] for r in e1['deletion']],marker=m,color=c,label=label)
    ax[0].axhline(100*e1['models']['mlp']['test'],ls='--',color='black',label='MLP on node features')
    ax[0].set(xlabel='Edge deletion probability p',ylabel='Test accuracy (%)',ylim=(0,100));ax[0].legend(fontsize=7)
    ax[1].plot(p,[r['isolated'] for r in e1['deletion']],'o-',label='Measured',color=colors[0])
    ax[1].plot(p,[r['expected_isolated'] for r in e1['deletion']],'s--',label='Sum p^degree',color=colors[1])
    ax[1].set(xlabel='Edge deletion probability p',ylabel='Isolated nodes',ylim=(0,2800));ax[1].legend()
    ax[2].plot(p,[r['hE'] if r['hE'] is not None else np.nan for r in e1['deletion']],'o-',color=colors[2])
    ax[2].set(xlabel='Edge deletion probability p',ylabel='Edge homophily hE',ylim=(0,1),xlim=(0,1));ax[2].text(.95,.1,'p=1: undefined',ha='right')
    save(fig,'E1_deletion','Shared seed1 edge draws; independent deletion per undirected edge. MLP has separately selected val hyperparameters.')
    fig,ax=plt.subplots(1,2,figsize=(10,4));seeds=[1,2,3]
    for name,label,c,m in [('original_test','GCN original',colors[0],'o'),('null_test','GCN degree-preserving null',colors[1],'s'),('MLP_test','MLP',colors[2],'^')]:
        ax[0].plot(seeds,[100*r[name] for r in paired],marker=m,label=label,color=c)
    ax[0].set(xlabel='Paired seed',ylabel='Test accuracy (%)',ylim=(0,100),xticks=seeds);ax[0].legend(fontsize=8)
    ax[1].scatter(seeds,[100*r['delta'] for r in paired],color=colors[0],s=50,label='Each paired difference')
    ax[1].axhline(0,color='black',lw=1);ax[1].axhline(100*e2['mean_delta'],ls='--',label='Mean difference',color=colors[1])
    ax[1].errorbar([3.5],[100*e2['mean_delta']],yerr=[100*e2['sample_sd_delta']],fmt='D',capsize=5,color=colors[1],label='Mean ± sample SD (3 seeds)')
    ax[1].set(xlabel='Seed / mean at right',ylabel='Original − null (percentage points)',xticks=seeds);ax[1].legend(fontsize=8)
    save(fig,'E2_paired','Three seeds on one fixed split; paired sample SD uses ddof=1. Degree sequence is exactly preserved.')
    fig,ax=plt.subplots(2,2,figsize=(11,8));depth=[r['depth'] for r in depths]
    ax[0,0].plot(depth,[100*r['train'] for r in depths],'o-',color=colors[0],label='Train, selected checkpoint')
    ax[0,0].plot(depth,[100*r['val'] for r in depths],'s-',color=colors[1],label='Val, selects depth')
    best=next(r for r in depths if r['depth']==e3['lstar_validation']);ax[0,0].axvline(best['depth'],ls=':',color='gray')
    ax[0,0].set(xlabel='GCN depth',ylabel='Accuracy (%)',ylim=(0,100),xticks=depth);ax[0,0].legend(fontsize=8)
    for i,(name,s) in enumerate(idx['smoothing'].items()):
        l=s['layers'];style='--' if name=='untrained' else '-';marker='x' if name=='untrained' else 'o';c=colors[i%len(colors)]
        for target,key,ylabel in [(ax[0,1],'energy','Normalized Dirichlet energy'),(ax[1,0],'probe_val','Val probe accuracy (%)'),(ax[1,1],'S','Mean pair cosine S')]:
            vals=[r[key] if r[key] is not None else np.nan for r in l]
            if key=='probe_val':vals=[100*v for v in vals]
            target.plot([r['layer'] for r in l],vals,linestyle=style,marker=marker,markersize=3,lw=1,label=name,color=c)
            target.set(xlabel='Layer (0 = input features)',ylabel=ylabel)
    ax[0,1].set_yscale('log');ax[1,0].set_ylim(0,100);ax[1,1].set_ylim(-.05,1.05)
    ax[0,1].legend(fontsize=7,ncol=2)
    save(fig,'E3_layers','Plain GCNs use E1 hyperparameters. Probes use train targets and val selection only; input/output layers are included.')
    fig,ax=plt.subplots(1,3,figsize=(11,4));plain=depths[-1];res=e3['residual']
    for target,key,label in [(ax[0],'test','Test accuracy (%)'),(ax[1],'probe_val','Output val probe (%)'),(ax[2],'energy','Output normalized energy')]:
        v=[plain[key],res[key]]
        if key!='energy':v=[100*x for x in v]
        target.bar(['Plain24','Residual24'],v,color=[colors[0],colors[1]],hatch=['','//']);target.set_ylabel(label)
        if key!='energy':target.set_ylim(0,100)
        for i,value in enumerate(v):target.text(i,value,f'{value:.3g}',ha='center',va='bottom',fontsize=9)
    save(fig,'E3_residual','One seed; hidden residual skips include a learned input projection. Output has no residual or ReLU.')
    # Structure map explicitly marks changes in more than one factor.
    rows=[]
    def add(condition,model,r,seed,depth,graph,comparison):
        rows.append({'condition':condition,'model':model,'graph':graph,'seed':seed,'depth':depth,'width':64,'dropout':.5,
          'train':r['train'],'val':r['val'],'test':r['test'],'RUN_ID':r['RUN_ID'],'source_RUN_ID':r['source_RUN_ID'],
          'fixed':'Cora X,y,public masks; transductive; Adam; full200 epochs; val best checkpoint',
          'comparison_scope':comparison})
    add('original','GCN',e1['models']['gcn'],1,2,'original','Direct: vary graph, GCN E1 lr/wd fixed')
    for pval in [.5,1]:
        key=f'E1_B_{pval}';add('half_deleted' if pval==.5 else 'empty','GCN',idx['evaluations'][key],1,2,f'delete_{pval}','Direct: vary graph, seed and architecture/hyperparameters fixed')
    add('degree_null','GCN',e2['paired'][0]['null'],1,2,'null_1','Direct graph intervention; seed1 E2 repeat of E1 original disclosed')
    add('ER','GCN',e2['ER'],1,2,'ER','Auxiliary: edge count fixed, degree distribution changes')
    for name,d in [('val_best_depth',e3['lstar_validation']),('maximum_depth',24)]:
        add(name,'GCN',next(r for r in e3['plain'] if r['depth']==d),1,d,'original','Direct depth intervention; no residual or normalization')
    add('feature_only','MLP',e1['models']['mlp'],1,2,'ignored','Auxiliary: architecture and separately val-selected hyperparameters change')
    add('maximum_residual','ResidualGCN',res,1,24,'original','Auxiliary: architecture changes; projected hidden skips')
    table('structure_map.csv',rows);write_json(R/'structure_map.json',rows)
    # All counterfactual text is generated from measured results and explicit RUN references.
    null=e2['paired'][0]['null'];mlp=e2['paired'][0]['mlp'];gat=e3['CF1_GAT_null'];added=e3['CF2_added'];prediction=read(R/'CF2_prediction.json')
    half=next(r for r in e1['deletion'] if r['p']==.5);base=e1['models']['gcn'];measure=added['graph_statistics']['hE']
    witnesses=[]
    for name,s in idx['smoothing'].items():
        if name=='untrained':continue
        bestprobe=max(r['probe_val'] for r in s['layers'])
        for l in s['layers']:
            if l['S']>=.8 and l['probe_val']>=bestprobe-.02:witnesses.append({'condition':name,'analysis_RUN_ID':s['RUN_ID'],'best_probe':bestprobe,**l})
    cf4={'status':'PASS' if witnesses else 'FAIL_NO_WITNESS','predeclared_rule':'S>=.8; probe within .02 absolute of same model best layer probe','witnesses':witnesses}
    write_json(R/'CF4_witnesses.json',cf4)
    witness=witnesses[0] if witnesses else None
    cf4text=(f"实测 {witness['condition']} 第 {witness['layer']} 层：S={witness['S']:.6f}，Ê={witness['energy']:.6g}，probe={pct(witness['probe_val'])}，该模型最佳层 probe={pct(witness['best_probe'])}。分析 RUN_ID `{witness['analysis_RUN_ID']}`，probe RUN_ID `{witness['probe_RUN_ID']}`。" if witness else '没有层满足事前规定的 S≥.8 且 probe 距同模型最佳层≤2个百分点；保留 FAIL_NO_WITNESS，不更改阈值。')
    text=f'''# Lab5 counterfactual answers — measured evidence

All conditions are transductive; PyG public masks were user-authorized. The teacher split remains independently UNVERIFIED. Test outcomes below are descriptive, never hyperparameter/depth selection input.

## Q1 — degree-preserving null versus MLP

Seed1: MLP {pct(mlp['test'])} (`{mlp['RUN_ID']}`), null GCN {pct(null['test'])} (`{null['RUN_ID']}`), null GAT {pct(gat['test'])} (`{gat['RUN_ID']}`). Three-seed MLP and null GCN values are in E2_pairs.csv. In (5.4), self-feature coefficient is 1/(di+1), while neighbor j contributes 1/sqrt((di+1)(dj+1)). Random neighbors can dilute class-relevant own features. For the measured degrees 1,3,168 these self weights are .5,.25,1/169 respectively. GAT (5.6) learns normalized attention including the self loop; it may reduce neighbor interference but has no guarantee of entirely rejecting random edges. The measured GAT−nullGCN difference is {100*(gat['test']-null['test']):.2f}pp, not evidence that a random graph contains useful class edges. Null preserves degree information; model/hyperparameter differences also prevent treating MLP−GCN as a pure structural intervention.

## Q2 — predicted addition versus deletion

Before generating the addition graph, CF2_prediction.json recorded hE={prediction['hE']:.6f}, h0={prediction['h0']:.6f}, E={prediction['E']}, added={prediction['added']}, predicted new hE={prediction['predicted_hE']:.6f} by weighted mixture. The actual hE is {measure:.6f}, error {measure-prediction['predicted_hE']:+.6f}; finite nonedge sampling need not equal the class-frequency approximation. Added-graph GCN test={pct(added['test'])} (`{added['RUN_ID']}`). Retrained deletion p=.5 test={pct(half['B_test'])} (`{half['B_RUN_ID']}`). Addition minus deletion is {100*(added['test']-half['B_test']):+.2f}pp: {'addition is more damaging' if added['test']<half['B_test'] else 'deletion is more damaging'} in this fixed-seed comparison. The graph perturbations are different interventions, not an uncertainty estimate.

## Q3 — two deletion modes at p=.5

Original GCN={pct(base['test'])}. ModeA={pct(half['A_test'])}, drop={100*(base['test']-half['A_test']):.2f}pp (`{half['A_RUN_ID']}`); modeB={pct(half['B_test'])}, drop={100*(base['test']-half['B_test']):.2f}pp (`{half['B_RUN_ID']}`). {half['isolated_test']}/1000 test nodes become isolated. Small loss would not establish that the model ignores the graph: surviving neighbors remain homophilic and node features/self loops remain available. ModeA tests distribution shift; modeB permits parameter adaptation on that same damaged graph.

## Q4 — high cosine alone is insufficient

{cf4text} ReLU gives a shared positive direction and cosine is insensitive to vector magnitude and small discriminative directions. Energy measures degree-corrected neighbor variation relative to total signal; the train→val probe measures retained label-relevant information. High S without lost probe accuracy does not establish harmful oversmoothing. Probe uncertainty and this one fixed split still limit the conclusion.

## Q5 — posthoc forbidden selection comparison

The immutable all_training_selection_lock.json selected depth {e3['lstar_validation']} using val before E3 depth-test inspection. Posthoc test argmax is depth {e3['test_argmax_posthoc']}; {'they agree' if e3['lstar_validation']==e3['test_argmax_posthoc'] else 'they differ'}. The per-depth evaluation RUN_IDs are in E3_depths.csv. This posthoc computation changes no configuration, checkpoint or result. E1 had already exposed the separately required depth2 baseline test outcome; no claim of a wholly unseen depth2 test is made. Transduction permits X and edge access, not test-label selection. Selecting depth on test would reuse the evaluation labels and bias the final estimate even if the two argmax depths happen to agree.
'''
    text+='\nFor Q2, random added cross-class neighbors reduce edge homophily and can inject misleading messages. Random deletion largely preserves hE but reduces the available neighbors; the interventions damage different information. The measured one-seed difference does not quantify sampling uncertainty.\n'
    (R/'counterfactual_answers.md').write_text(text,encoding='utf-8')
    write_json(R/'figure_provenance.json',{'generator':'reports.tools.lab5_artifacts',
        'data_sources':{p.name:file_hash(p) for p in [R/'E1.json',R/'E2.json',R/'E3.json',R/'run_index.json']},
        'figures':{p.name:file_hash(p) for p in F.iterdir() if p.is_file()},
        'RUN_ID_sources':'E1_models.csv, E1_deletion.csv, E2_pairs.csv, E3_depths.csv, E3_all_layers.csv; source/evaluation/probe IDs all preserved'})
    print(json.dumps({'figures':4,'CF4':cf4['status'],'witnesses':len(witnesses),'posthoc_test_depth':e3['test_argmax_posthoc']}))
if __name__=='__main__':main()
