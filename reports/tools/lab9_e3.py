import numpy as np,torch
from scipy.stats import rankdata
from reports.tools.lab9_common import *
from src.multimodal.pairs import make_pairs,require_defined_metric

def auroc(labels,scores):
    y=np.asarray(labels,dtype=bool);r=rankdata(scores);n=y.sum();m=(~y).sum()
    if not n or not m:raise ValueError('AUROC requires both classes')
    return float((r[y].sum()-n*(n+1)/2)/(n*m))

def main():
    selection=read(R/'E1_validation_selection.json');specs=selection['specs'];ds=features()['test'];x=ds['x'];byid={c['clip_id']:c for c in read(R/'data_manifest.json')['clips']};clips=[byid[cid] for cid in ds['clip_ids']]
    pairs={p:make_pairs(clips,p,count=300,seed=1) for p in ['A','B']};write_json(R/'E3_pairs.json',pairs);results={}
    for name in ['early','gated','late05']:
        spec=specs[name];original,originalgate=predict(spec,x);conditions={};raw={'original_probabilities':original,'original_gates':originalgate,'original_clip_ids':ds['clip_ids']}
        for protocol,rows in pairs.items():
            vi=torch.tensor([r['video_index'] for r in rows]);ai=torch.tensor([r['audio_index'] for r in rows]);px=x[vi].clone();px[:,512:640]=x[ai,512:640]
            prob,g=predict(spec,px);noaudio,_=predict(spec,px,(1,0));pred=prob.argmax(1);yv=ds['y'][vi];ya=ds['y'][ai];deltaij=(prob-original[vi]).abs().sum(1);deltaa=(prob-noaudio).abs().sum(1)
            raw[protocol]={'probabilities':prob,'gates':g,'delta_ij':deltaij,'delta_a':deltaa,'video_indices':vi,'audio_indices':ai,'no_audio_probabilities':noaudio,'pair_ids':[r['video_clip_id']+'__'+r['audio_clip_id'] for r in rows]}
            cond={'N':len(rows),'confidence_mean':float(prob.max(1).values.mean()),'gate_mean':float(g.mean()) if g is not None else None,'gate_status':'DEFINED' if g is not None else 'NOT_APPLICABLE','delta_ij_mean':float(deltaij.mean()),'delta_a_mean':float(deltaa.mean())}
            if protocol=='A':cond.update(classification(prob.numpy(),yv.numpy()))
            else:
                rv=float((pred==yv).float().mean());ra=float((pred==ya).float().mean());r0=float(((pred!=yv)&(pred!=ya)).float().mean());assert abs(rv+ra+r0-1)<1e-6
                cond.update(rv=rv,ra=ra,r0=r0,undefined=[{'metric':m,'reason':'Conflicting labels; no unique true emotion'} for m in ['accuracy','macro_f1','ECE','selective_risk']])
                for metric in ['accuracy','macro_f1','ece','risk']:
                    try:require_defined_metric('B',metric)
                    except ValueError:pass
                    else:raise AssertionError('Protocol B fabricated labelled metric')
            conditions[protocol]=cond
        y=np.r_[np.zeros(300),np.ones(300)];confidence=np.r_[raw['A']['probabilities'].max(1).values.numpy(),raw['B']['probabilities'].max(1).values.numpy()];delta=np.r_[raw['A']['delta_a'].numpy(),raw['B']['delta_a'].numpy()]
        path=new_evaluation('E3_pairs_'+name,spec,{'pairs':pairs,'feature_sha256':file_hash(FEATURE)},seed=1);torch.save(raw,path/'predictions.pt')
        out={'RUN_ID':path.name,'strategy':name,'conditions':conditions,'original':{'confidence_mean':float(original.max(1).values.mean()),'gate_mean':float(originalgate.mean()) if originalgate is not None else None},'AUROC_1_minus_conf':auroc(y,1-confidence),'AUROC_delta_a':auroc(y,delta),'AUROC_random_baseline':.5,'limitations':'Different-actor pairing changes identity and emotion correspondence jointly. Five test actors, one split/seed; detection is not isolated causal evidence of semantic inconsistency. Delta_ij compares original pair; delta_a is the same current pair versus audio removal.'}
        out['fusion_profile']={'macro_f1_protocol_A':conditions['A']['macro_f1'],'following_protocol_B':{k:conditions['B'][k] for k in ['rv','ra','r0']},'undefined':conditions['B']['undefined']}
        write_json(path/'results.json',out);write_json(path/'summary.json',out);results[name]=out
    write_json(R/'E3.json',results)

if __name__=='__main__':main()
