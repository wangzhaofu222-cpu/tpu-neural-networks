"""Exploration -> validation lock -> external real Git preregistration -> paired main."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from reports.tools.lab9_common import *
from src.multimodal.fusion import parameter_count

def explore():
    assert (R/'exploratory_hypotheses.md').exists()
    ds=features();specs={};counts={};val={};tests={};controls={}
    for strategy in ['video','audio','early','gated']:
        path=exploratory_train('E1_explore_'+strategy,strategy);spec={'path':path,'strategy':strategy};specs[strategy]=spec
        model,_,_=load(path);counts[strategy]=training_capacity(model);counts[strategy]['external_frozen_parameters']=read(R/'feature_cache.json')['frozen_parameters']
        val[strategy]=classification(predict(spec,ds['val']['x'])[0].numpy(),ds['val']['y'].numpy())['macro_f1']
    pv=predict(specs['video'],ds['val']['x'])[0];pa=predict(specs['audio'],ds['val']['x'])[0]
    grid=[{'weight':k/10,'macro_f1':classification((k/10*pv+(1-k/10)*pa).numpy(),ds['val']['y'].numpy())['macro_f1']} for k in range(11)]
    weight=max(grid,key=lambda a:(a['macro_f1'],-abs(a['weight']-.5),-a['weight']))['weight']
    for key,w in [('late05',.5),('late_val',weight)]:
        specs[key]={'strategy':'late','video_path':specs['video']['path'],'audio_path':specs['audio']['path'],'weight':w}
        counts[key]={'total_trainable':counts['video']['total_trainable']+counts['audio']['total_trainable'],'fusion_trainable':0,'head_trainable':counts['video']['total_trainable']+counts['audio']['total_trainable'],'budget':'two constituent unimodal heads, each200epochs; cached merge has zero updates'}
        counts[key]['active_nonzero_weight_parameters']=sum(counts[s]['total_trainable'] for s,active in [('video',w>0),('audio',w<1)] if active)
        counts[key]['capacity_note']='Matched target counts both trained constituent heads. At lambda0/1 one branch has zero functional contribution; effective active capacity is separately recorded.'
        val[key]=classification(predict(specs[key],ds['val']['x'])[0].numpy(),ds['val']['y'].numpy())['macro_f1']
    for key in ['early','gated','late05','late_val']:
        target=counts[key]['total_trainable'];controls[key]={}
        for strategy in ['video','audio']:
            match=matched_control(target,strategy);name=f'{strategy}_plus_{target}';existing=next((v for k,v in specs.items() if k==name),None)
            if existing is None:
                path=exploratory_train('E1_explore_'+name,strategy,width=match['width']);specs[name]={'path':path,'strategy':strategy}
                m,_,_=load(path);counts[name]=training_capacity(m);assert counts[name]['total_trainable']==match['actual']
                val[name]=classification(predict(specs[name],ds['val']['x'])[0].numpy(),ds['val']['y'].numpy())['macro_f1']
            controls[key][strategy]={'name':name,'match':match}
    # Lock before reading any E1 test predictions. Tie-breaking deterministic registry order.
    F=max(['early','gated','late05','late_val'],key=lambda k:val[k]);candidates=[controls[F][s]['name'] for s in ['video','audio']];U=max(candidates,key=lambda k:val[k])
    selection={'criterion':'validation macro-F1 only','F':F,'U':U,'validation':val,'late_weight_grid':grid,'late_weight':weight,'controls':controls,'parameter_counts':counts,'specs':specs,'features_sha256':file_hash(FEATURE),'seeds_main':[1,2,3],'retrain_seed1_after_prereg':True}
    write_json(R/'E1_validation_selection.json',selection)
    for key,spec in specs.items():tests[key]=evaluate('E1_explore_test_'+key,spec)
    gains={}
    for key in ['early','gated','late05','late_val']:
        v=tests[key]['macro_f1'];gains[key]={'G':v-max(tests['video']['macro_f1'],tests['audio']['macro_f1']),'Gint':v-tests['late_val']['macro_f1'],'Gctrl':v-max(tests[controls[key][s]['name']]['macro_f1'] for s in ['video','audio'])}
        tests[key]['fusion_profile']['gains']=gains[key];tests[key]['fusion_profile']['parameter_breakdown']=counts[key]
        evaluation=Path('runs/lab9')/tests[key]['RUN_ID'];write_json(evaluation/'summary.json',tests[key]);write_json(evaluation/'results.json',tests[key])
    write_json(R/'E1_exploration.json',{'selection':selection,'test':tests,'gains':gains})
    hypothesis=(f'# E1 主比较事前假设 / Предварительная гипотеза\n\nValidation-selected F={F}, U={U}. Direction Δ=macroF1(F)−macroF1(U). Predict Δ>0: complementary speech/face cues may improve over parameter-matched unimodal capacity. This is a hypothesis, not an established result.\n\nExploration seed1 has already been observed. All main seeds1,2,3 will be retrained after this Git registration; this does not make exploration seed1 unseen. Shared frozen encoders, actor split, train-only normalization, input features SHA256 '+selection['features_sha256']+', batch64, AdamW lr.001/wd.0001, 200epochs per trained head, validation-best macroF1, no head early stop, fixed evaluation masks/labels. Only F versus its capacity-matched unimodal U changes. Late fusion, if selected, requires two unimodal constituents each200epochs; its extra training cost is disclosed, not claimed compute-matched.\n\nFalsification: Δ mean≤0 rejects predicted sign; mixed signs or |mean|<2 sampleSD means различие не установлено even if mean positive. Reverse established effects remain prediction FAIL. Metric noise is not removed through seed filtering. No test selection or seed replacement.\n\nSelection evidence: E1_validation_selection.json. Main results must refer to the Git commit containing this unchanged file.\n')
    (R/'hypothesis.md').write_text(hypothesis,encoding='utf-8');print('EXPLORATION COMPLETE; COMMIT hypothesis BEFORE main',F,U,flush=True)

def main_experiment():
    import subprocess
    commit=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','rev-parse','HEAD'],text=True).strip()
    blob=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show',commit+':experiments/lab9/hypothesis.md'])
    assert blob.decode('utf-8').replace('\r\n','\n')==(R/'hypothesis.md').read_text(encoding='utf-8')
    assert not git_state()['dirty'];selection=read(R/'E1_validation_selection.json');F=selection['F'];U=selection['U'];rows=[]
    write_json(R/'E1_preregistration.json',{'commit':commit,'hypothesis_sha256':file_hash(R/'hypothesis.md'),'selection_sha256':file_hash(R/'E1_validation_selection.json'),'before_main_training':True})
    for seed in [1,2,3]:
        if selection['specs'][F]['strategy']=='late':
            v=train(f'E1_main_F_video_seed{seed}','video',seed);a=train(f'E1_main_F_audio_seed{seed}','audio',seed);fs={'strategy':'late','video_path':v,'audio_path':a,'weight':selection['specs'][F]['weight']}
        else:fs={'path':train(f'E1_main_F_seed{seed}',F,seed),'strategy':F}
        strategy=selection['specs'][U]['strategy'];w=selection['controls'][F][strategy]['match']['width'];us={'path':train(f'E1_main_U_seed{seed}',strategy,seed,width=w),'strategy':strategy}
        fr=evaluate(f'E1_main_F_test_seed{seed}',fs,seed=seed);ur=evaluate(f'E1_main_U_test_seed{seed}',us,seed=seed)
        rows.append({'seed':seed,'F':fr,'U':ur,'delta':fr['macro_f1']-ur['macro_f1']});write_json(R/'E1_main_progress.json',rows)
    d=np.array([r['delta'] for r in rows]);mean=float(d.mean());sd=float(d.std(ddof=1));same=bool((d>0).all() or (d<0).all());est=same and abs(mean)>=2*sd
    write_json(R/'E1_main.json',{'F':F,'U':U,'rows':rows,'differences':d.tolist(),'mean':mean,'sample_std':sd,'all_same_sign':same,'abs_mean_ge_2sd':abs(mean)>=2*sd,'decision':'различие установлено' if est else 'различие не установлено','prediction':'PASS' if est and mean>0 else ('FAIL' if mean<=0 else 'NOT_ESTABLISHED'),'preregistration_commit':commit})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['explore','main']);a=p.parse_args();explore() if a.stage=='explore' else main_experiment()
