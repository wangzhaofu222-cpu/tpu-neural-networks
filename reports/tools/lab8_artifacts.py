"""Offline tables/figures only: no model inference, fit, attack or training."""
import csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reports.tools.lab8_common import read,outputs
from src.reliability.uncertainty import uncertainty
from src.utils.io import write_json,file_hash

R=Path('experiments/lab8');F=R/'figures'
plt.rcParams.update({'font.size':9,'svg.hashsalt':'lab8-v1','axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.tight_layout();fig.savefig(F/(name+'.png'),dpi=160);fig.savefig(F/(name+'.svg'),metadata={'Date':None});plt.close(fig)
def table(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (R/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def flat(row):
    return {**{k:v for k,v in row.items() if k!='ece'},'ECE':row['ece']['value'],'ECE_M':row['ece']['M'],'ECE_binning':row['ece']['binning'],'ECE_empty_bins':row['ece']['empty_bins']}

def main():
    F.mkdir(parents=True,exist_ok=True);e1=read(R/'E1.json');idx=read(R/'run_index.json')
    table('E1.csv',[flat(r) for r in e1['rows']]);table('E1_binning.csv',e1['binning_sensitivity'])
    for arch in ['cnn','vit']:
        rows=[r for r in e1['rows'] if r['model']==arch];fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for r in rows:
            bins=[b for b in r['ece']['bins'] if b['count']];T=r['temperature'];axes[0].plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],'.-',label=f'T={T:.3f}, ECE={r["ece"]["value"]:.3f}')
            d=outputs(idx[f'clean_test_{arch}_1']);c=(d['logits'].double()/T).softmax(1).max(1).values.numpy();axes[1].hist(c,bins=np.linspace(0,1,16),histtype='step',label=f'T={T:.3f}')
        axes[0].plot([0,1],[0,1],'k--',lw=1);axes[0].set(xlabel='Mean confidence',ylabel='Accuracy in bin',xlim=(0,1),ylim=(0,1));axes[1].set(xlabel='Confidence',ylabel='Objects')
        for ax in axes:ax.legend(fontsize=8)
        fig.suptitle(f'{arch.upper()}: E1, M=15 equal-width, clean test N=10000');save(fig,'E1_'+arch)
    fig,ax=plt.subplots(figsize=(8,4))
    for arch in ['cnn','vit']:
        rows=[r for r in e1['rows'] if r['model']==arch];vals=[]
        for M,b in [(5,'equal_width'),(15,'equal_width'),(50,'equal_width'),(15,'equal_count')]:
            if M==15 and b=='equal_width':vals.append([r['ece']['value'] for r in rows])
            else:vals.append([r['value'] for r in e1['binning_sensitivity'] if r['model']==arch and r['M']==M and r['binning']==b])
        ax.plot(range(4),np.array(vals)[:,0],'.--',label=arch+' T=1');ax.plot(range(4),np.array(vals)[:,1],'.-',label=f'{arch} Tstar={rows[1]["temperature"]:.3f}')
    ax.set(xticks=range(4),xticklabels=['M5 width','M15 width','M50 width','M15 equal count'],ylabel='ECE',title='E1 binning sensitivity, clean test');ax.legend(loc='center',bbox_to_anchor=(.5,.68));save(fig,'E1_binning')
    if (R/'E2.json').exists():
        e2=read(R/'E2.json');table('E2_grid.csv',[flat(r) for r in e2['rows']]);table('E2_pairs.csv',e2['pairs']);table('E2_T5.csv',[{'condition':k,'RUN_ID':e2['T5'][k]['RUN_ID'],**flat(m)} for k in ['clean','shift5'] for m in e2['T5'][k]['metrics']])
        for kind in ['gaussian_noise','gaussian_blur','brightness','contrast','occlusion']:
            rr=[r for r in e2['rows'] if r['seed']==1 and r['corruption']==kind];fig,axes=plt.subplots(2,3,figsize=(12,7));Ts=list(dict.fromkeys(r['temperature'] for r in rr))
            for T in Ts:
                a=[r for r in rr if r['temperature']==T]
                for ax,key in zip(axes.flat,['accuracy','mean_confidence','ece','nll','brier']):ax.plot([r['level'] for r in a],[r['ece']['value'] if key=='ece' else r[key] for r in a],'.-',label=f'T={T:.3f}');ax.set(xlabel='Severity',ylabel=key)
            gains=e2['boundaries'][kind]['gains'];axes[1,2].plot(range(6),gains,'.-');axes[1,2].axhline(0,color='k',lw=.7);axes[1,2].set(xlabel='Severity',ylabel='g = ECE1 - ECE(Tstar)')
            last=next(r for r in rr if r['level']==5 and r['temperature']==1.)
            axes[0,1].text(.03,.03,f"At level5: accuracy={last['accuracy']:.3f}\nconfidence-accuracy={last['mean_confidence']-last['accuracy']:.3f}",transform=axes[0,1].transAxes,fontsize=8)
            axes[0,0].legend();fig.suptitle(f'CNN seed1: {kind}; M=15 equal-width; local explicit operator');save(fig,'E2_'+kind)
        fig,ax=plt.subplots(figsize=(6.8,4));ax.plot([1,2,3],[p['gA'] for p in e2['pairs']],'.-',label='clean gA');ax.plot([1,2,3],[p['gB'] for p in e2['pairs']],'.-',label='occlusion5 gB');ax.plot([1,2,3],e2['statistics']['deltas'],'s--',label='Delta=gA-gB');ax.axhline(0,color='k',lw=.7);ax.set(xlabel='Genuine training seed',ylabel='ECE reduction',xticks=[1,2,3],title='E2 paired comparison, M=15 equal-width');ax.legend();save(fig,'E2_pairs')
    if (R/'E3.json').exists():
        e3=read(R/'E3.json');table('E3.csv',[{'source_key':r['source_key'],'attack':r['attack'],'eps_255':r['eps_255'],'K':r.get('K'),'R':r.get('R'),'RUN_ID':r['RUN_ID'],**flat(m)} for r in e3['rows'] for m in r['metrics']])
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for arch in ['cnn','vit']:
            rows=[r for r in e3['rows'] if r['source_key']==arch+'_1'];fg=[r for r in rows if r['attack']=='FGSM']
            color='tab:blue' if arch=='cnn' else 'tab:orange'
            for ax,key in zip(axes,['accuracy','mean_confidence']):
                ax.plot([r['eps_255'] for r in fg],[r['metrics'][0][key] for r in fg],'.-',color=color,label=arch+' FGSM T1')
                for r in rows:
                    if r['attack']=='PGD':ax.scatter(r['eps_255'],r['metrics'][0][key],color=color,marker='x' if r['K']==10 else 's',label=f'{arch} PGD K{r["K"]} R{r["R"]}')
                ax.set(xlabel='epsilon ×255',ylabel=key);ax.legend(fontsize=7)
        fig.suptitle('E3: full shared test, raw pixels [0,1], PGD alpha=2.5 epsilon/K');save(fig,'E3_attacks')
        import pickle
        with open('data/cifar-10-batches-py/test_batch','rb') as f:ds=pickle.load(f,encoding='bytes')
        base=ds[b'data'].reshape(-1,3,32,32);split=read('splits/cifar10.json')['roles']['test']
        for arch in ['cnn','vit']:
            fg=Path(idx[f'FGSM_{arch}_1_2']);pg=Path(idx[f'PGD_{arch}_1_10_1']);a=torch.load(fg/'batches/000000.pt',weights_only=False)['adversarial_pixels'];b=torch.load(pg/'batches/000000.pt',weights_only=False)['adversarial_pixels'];x=torch.tensor(base[split[:4]].copy()).float()/255
            fig,axes=plt.subplots(4,4,figsize=(8,8))
            for i in range(4):
                delta=b[i]-x[i]
                for j,v in enumerate([x[i],a[i],b[i],(.5+delta*255/4).clamp(0,1)]):axes[i,j].imshow(v.permute(1,2,0));axes[i,j].axis('off')
                axes[i,3].set_title(f'Linf={float(delta.abs().max()):.5f}',fontsize=8)
            for j,title in enumerate(['Clean','FGSM 2/255','PGD10 R1 2/255','0.5 + delta ×63.75']):axes[0,j].set_title(title,fontsize=8)
            fig.suptitle(f'{arch.upper()}: identical test objects, alpha=0.25 epsilon');save(fig,'E3_examples_'+arch)
    if (R/'selective_ood.json').exists():
        ss=read(R/'selective_ood.json');table('selective_ood.csv',[{'source_key':r['source_key'],'RUN_ID':r['RUN_ID'],'temperature':c['temperature'],'measure':k,**{q:v for q,v in v['selective'].items() if q not in ['coverage','risk']},**v['ood']} for r in ss for c in r['conditions'] for k,v in c['measures'].items()])
        for r in ss:
            sk=r['source_key'];fig,axes=plt.subplots(1,2,figsize=(11,4));td=outputs(idx['clean_test_'+sk]);od=outputs(idx['OOD_'+sk]);T=r['conditions'][1]['temperature']
            for c in r['conditions']:
                for measure,v in c['measures'].items():axes[0].plot(v['selective']['coverage'],v['selective']['risk'],lw=1,label=f'{measure}, T={c["temperature"]:.2f}')
            oracle=r['conditions'][0]['measures']['msp']['selective']['oracle_AURC'];axes[0].set(xlabel='Coverage',ylabel='Risk',title=f'Clean test, oracle AURC={oracle:.4f}');axes[0].legend(fontsize=7)
            # Equal-bin density histograms, three uncertainty measures separately below.
            iu=uncertainty(td['logits'],T)['msp'].numpy();ou=uncertainty(od['logits'],T)['msp'].numpy();axes[1].hist(iu,bins=np.linspace(0,1,41),density=True,histtype='step',label='CIFAR10 ID');axes[1].hist(ou,bins=np.linspace(0,1,41),density=True,histtype='step',label='SVHN OOD');axes[1].set(xlabel='U_MSP',ylabel='Density',title=f'Tstar={T:.3f}, OOD positive');axes[1].legend();fig.suptitle(sk+': selective prediction and far OOD');save(fig,'selective_ood_'+sk)
            fig,axes=plt.subplots(1,3,figsize=(12,3.6))
            for ax,measure in zip(axes,['msp','entropy','margin']):
                iu=uncertainty(td['logits'],T)[measure].numpy();ou=uncertainty(od['logits'],T)[measure].numpy();bins=np.linspace(min(iu.min(),ou.min()),max(iu.max(),ou.max()),41)
                ax.hist(iu,bins=bins,density=True,histtype='step',label='ID');ax.hist(ou,bins=bins,density=True,histtype='step',label='OOD');ax.set(xlabel=measure,ylabel='Density');ax.legend()
            fig.suptitle(f'{sk}: full SVHN test, Tstar={T:.3f}; higher U = more uncertain');save(fig,'OOD_measures_'+sk)
    names=['E1.json','E2.json','E3.json','selective_ood.json'];write_json(R/'figure_provenance.json',{'source_records':{n:file_hash(R/n) for n in names if (R/n).exists()},'figure_groups':len(list(F.glob('*.png'))),'figure_hashes':{p.name:file_hash(p) for p in F.glob('*')}})
    print('Figure groups',len(list(F.glob('*.png'))),flush=True)

if __name__=='__main__':main()
