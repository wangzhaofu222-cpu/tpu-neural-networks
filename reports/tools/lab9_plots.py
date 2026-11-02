"""Rebuild tables and scientific figures exclusively from saved raw results."""
import csv,json
from pathlib import Path
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reports.tools.lab9_common import R,read
from src.utils.io import write_json,file_hash

def main(destination=None):
    out=Path(destination) if destination else R/'figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'lab9','figure.dpi':120})
    made=[]
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','svg']:
            p=out/(name+'.'+ext);fig.savefig(p,metadata={'Date':None} if ext=='svg' else {'Software':'Lab9 reproducible plots'});made.append(p)
        plt.close(fig)
    e1=read(R/'E1_exploration.json');e2=read(R/'E2.json');e3=read(R/'E3.json');main=read(R/'E1_main.json')
    keys=list(e1['test']);rows=[]
    for k in keys:
        r=e1['test'][k];rows.append({'model':k,'RUN_ID':r['RUN_ID'],'val_macro_f1':e1['selection']['validation'][k],'test_macro_f1':r['macro_f1'],'accuracy':r['accuracy'],'parameters':e1['selection']['parameter_counts'][k]['total_trainable'],**e1['gains'].get(k,dict(G='',Gint='',Gctrl=''))})
    with (out/'E1.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(11,4));ax.bar(np.arange(len(keys)),[r['test_macro_f1'] for r in rows]);ax.set(xticks=np.arange(len(keys)),xticklabels=keys,ylabel='Test macro-F1',ylim=(0,1));ax.tick_params(axis='x',rotation=45);save(fig,'E1_models')
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for ax,k in zip(axs.flat,['video','audio','early','gated']):
        im=ax.imshow(e1['test'][k]['confusion_matrix'],cmap='Blues');ax.set(title=k,xlabel='Prediction',ylabel='True emotion');fig.colorbar(im,ax=ax)
    save(fig,'E1_confusion')
    fig,ax=plt.subplots(figsize=(6,4))
    for row in main['rows']:ax.plot([0,1],[row['U']['macro_f1'],row['F']['macro_f1']],marker='o',label=f"seed{row['seed']}")
    ax.set(xticks=[0,1],xticklabels=[main['U'],main['F']],ylabel='Test macro-F1');ax.legend();save(fig,'E1_paired')
    fig,ax=plt.subplots(figsize=(9,4));models=list(e2['models']);xx=np.arange(len(models))
    for i,(label,field) in enumerate([('Full','M_AV'),('No audio','M_without_audio'),('No video','M_without_video')]):ax.bar(xx+(i-1)*.25,[e2['models'][m]['contribution'][field] for m in models],width=.25,label=label)
    ax.set(xticks=xx,xticklabels=models,ylabel='Macro-F1',ylim=(0,1));ax.legend();save(fig,'E2_missing')
    fig,axs=plt.subplots(1,2,figsize=(10,4));noise_rows=[]
    for model,rs in e2['noise'].items():
        for r in rs:noise_rows.append({'model':model,'SNR':r['snr_db'],'macro_f1':r['macro_f1'],'gate_video_weight':r['gate']['mean'],'RUN_ID':r['RUN_ID']})
        axs[0].plot([r['snr_db'] for r in rs],[r['macro_f1'] for r in rs],marker='o',label=model);axs[0].axhline(e2['models'][model]['contribution']['M_without_audio'],linestyle='--',alpha=.5)
        axs[1].plot([r['snr_db'] for r in rs],[r['gate']['mean'] for r in rs],marker='o',label=model)
    for ax in axs:ax.set_xlabel('SNR (dB)');ax.legend();ax.invert_xaxis()
    axs[0].set_ylabel('Macro-F1 (dashed: no audio)');axs[1].set_ylabel('Mean gate (1 = video)');save(fig,'E2_SNR')
    with (out/'E2_SNR.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(noise_rows[0]));w.writeheader();w.writerows(noise_rows)
    fig,axs=plt.subplots(1,3,figsize=(12,3.7))
    for ax,(name,r) in zip(axs,e3.items()):
        raw=torch.load(Path('runs/lab9')/r['RUN_ID']/'predictions.pt',weights_only=False)
        ax.hist(raw['original_probabilities'].max(1).values.numpy(),bins=np.linspace(0,1,21),alpha=.4,density=True,label='Original')
        for p in ['A','B']:ax.hist(raw[p]['probabilities'].max(1).values.numpy(),bins=np.linspace(0,1,21),alpha=.4,density=True,label=p)
        ax.set(title=name,xlabel='Confidence',ylabel='Density');ax.legend()
    save(fig,'E3_confidence')
    gated=e3['gated'];raw=torch.load(Path('runs/lab9')/gated['RUN_ID']/'predictions.pt',weights_only=False);fig,ax=plt.subplots(figsize=(6,4))
    for label,g in [('Original',raw['original_gates']),('A',raw['A']['gates']),('B',raw['B']['gates'])]:ax.hist(g.flatten().numpy(),bins=np.linspace(0,1,21),alpha=.4,label=label,density=True)
    ax.set(xlabel='Gate (1 = video)',ylabel='Density');ax.legend();save(fig,'E3_gates')
    fig,axs=plt.subplots(1,2,figsize=(10,4));names=list(e3);bottom=np.zeros(3)
    for k in ['rv','ra','r0']:
        values=np.array([e3[n]['conditions']['B'][k] for n in names]);axs[0].bar(names,values,bottom=bottom,label=k);bottom+=values
    axs[0].set_ylabel('Protocol B following fraction');axs[0].legend()
    xx=np.arange(3);axs[1].bar(xx-.18,[e3[n]['AUROC_1_minus_conf'] for n in names],width=.36,label='1-conf');axs[1].bar(xx+.18,[e3[n]['AUROC_delta_a'] for n in names],width=.36,label='delta_a');axs[1].axhline(.5,color='black',linestyle='--');axs[1].set(xticks=xx,xticklabels=names,ylabel='AUROC (A negative / B positive)',ylim=(0,1));axs[1].legend();save(fig,'E3_following_AUROC')
    # Learning curves come from shared-harness epoch logs, not reconstructed toy curves.
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for k in ['video','audio','early','gated']:
        path=Path(e1['selection']['specs'][k]['path']);records=[json.loads(l) for l in (path/'metrics.jsonl').read_text().splitlines()]
        axs[0].plot([r['epoch'] for r in records],[r['train_loss'] for r in records],label=k)
        axs[1].plot([r['epoch'] for r in records],[r['validation_metric'] for r in records],label=k)
    axs[0].set(xlabel='Epoch',ylabel='Train cross-entropy');axs[1].set(xlabel='Epoch',ylabel='Val macro-F1')
    for ax in axs:ax.legend()
    save(fig,'training_curves')
    write_json(out/'artifact_hashes.json',{p.name:file_hash(p) for p in sorted(out.iterdir()) if p.suffix in ['.csv','.png','.svg']});print('FIGURES',len(made),'TABLES',2,flush=True)

if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else None)
