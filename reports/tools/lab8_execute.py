"""Ordered Lab8 phases; E2 refuses to run without its real Git preregistration."""
import argparse,copy,time
from pathlib import Path
import numpy as np
import torch
from reports.tools.lab8_common import *
from src.reliability.calibration import TemperatureScaler
from src.reliability.selective import risk_coverage
from src.reliability.ood import detection_metrics
from src.utils.preregistration import verify_git_hypothesis

def fit_temperature(key,source_key,val_run):
    idx=registry()
    if key in idx and (Path(idx[key])/'summary.json').exists():return read(Path(idx[key])/'summary.json')
    d=outputs(val_run);_,_,allow=role_data('val');scaler=TemperatureScaler(allow).fit(d['logits'],d['labels'],d['object_IDs'])
    m,source=source_model(source_key);p,manifest=observation(key,{'fit_role':'val','input_RUN_ID':Path(val_run).name,'input_sha256':file_hash(Path(val_run)/'outputs.pt')},m,source)
    result={**scaler.fit_info,'RUN_ID':p.name,'val_logits_RUN_ID':Path(val_run).name};finish(p,manifest,result);return result

def metrics_at(p,T):
    d=outputs(p);return classification_metrics(d['logits'],d['labels'],temperature=T)

def calibrated_record(p,T):
    d=outputs(p);s=read(Path(p)/'summary.json')
    records(Path(p)/'records_Tstar.jsonl',d['logits'],d['labels'],d['object_IDs'],T,{'RUN_ID':Path(p).name,'model':s['source_key'],'seed':int(s['source_key'].split('_')[1]),'corruption':s['corruption'],'level':s['level']})

def e1():
    rows=[];temperatures={};sensitivity=[]
    for arch in ['cnn','vit']:
        sk=arch+'_1';vp=evaluate('clean_val_'+sk,sk,'val');tp=evaluate('clean_test_'+sk,sk)
        fit=fit_temperature('temperature_'+sk,sk,vp);T=fit['temperature'];temperatures[sk]=fit
        d=outputs(tp);calibrated_record(tp,T)
        for temp in [1.,T]:
            row={'model':arch,'seed':1,'RUN_ID':tp.name,'fit_RUN_ID':fit['RUN_ID'],**metrics_at(tp,temp)};rows.append(row)
            for M,binning in [(5,'equal_width'),(50,'equal_width'),(15,'equal_count')]:
                sensitivity.append({'model':arch,'RUN_ID':tp.name,'temperature':temp,**classification_metrics(d['logits'],d['labels'],temperature=temp,M=M,binning=binning)['ece']})
        assert torch.equal(d['logits'].argmax(1),(d['logits']/T).argmax(1))
    write_json(R/'temperatures.json',temperatures);write_json(R/'E1.json',{'rows':rows,'binning_sensitivity':sensitivity,'argmax_unchanged_fraction':1.})

def e2():
    evidence=read(R/'preregistration_evidence.json')
    assert verify_git_hypothesis(evidence['commit'],'experiments/lab8/hypothesis.md')
    profile=config()['profile'];arch=profile['lab8_arch'];kind=profile['lab8_corruption'];temps=read(R/'temperatures.json');rows=[];pairs=[]
    for seed in [1,2,3]:
        sk=f'{arch}_{seed}';vp=evaluate('clean_val_'+sk,sk,'val');tp=evaluate('clean_test_'+sk,sk)
        if sk not in temps:temps[sk]=fit_temperature('temperature_'+sk,sk,vp);write_json(R/'temperatures.json',temps)
        T=temps[sk]['temperature'];clean=[metrics_at(tp,t) for t in [1.,T]]
        calibrated_record(tp,T)
        # Exploratory complete grid uses seed 1; seed2/3 are genuine paired main repeats.
        conditions=[(c,l) for c in config()['corruptions'] for l in range(6)] if seed==1 else [(kind,5)]
        for c,l in conditions:
            p=tp if l==0 else evaluate(f'corruption_{sk}_{c}_{l}',sk,kind=c,level=l)
            metrics=[metrics_at(p,t) for t in [1.,T]]
            for metric in metrics:rows.append({'source_key':sk,'seed':seed,'corruption':c,'level':l,'RUN_ID':p.name,'fit_RUN_ID':temps[sk]['RUN_ID'],**metric})
            if l:calibrated_record(p,T)
            if c==kind and l==5:
                gA=clean[0]['ece']['value']-clean[1]['ece']['value'];gB=metrics[0]['ece']['value']-metrics[1]['ece']['value']
                pairs.append({'seed':seed,'clean_RUN_ID':tp.name,'shift_RUN_ID':p.name,'temperature':T,'gA':gA,'gB':gB,'delta':gA-gB,'clean':clean,'shift':metrics})
        write_json(R/'E2_progress.json',{'rows':rows,'pairs':pairs})
    vp5=evaluate('T5_val_'+arch+'_1',arch+'_1','val',kind=kind,level=5);fit5=fit_temperature('temperature5_'+arch+'_1',arch+'_1',vp5)
    T5=fit5['temperature'];T=temps[arch+'_1']['temperature'];cleanp=Path(registry()['clean_test_'+arch+'_1']);shiftp=Path(registry()[f'corruption_{arch}_1_{kind}_5'])
    diagnostic={'fit':fit5,'clean':{'RUN_ID':cleanp.name,'metrics':[metrics_at(cleanp,t) for t in [1.,T,T5]]},'shift5':{'RUN_ID':shiftp.name,'metrics':[metrics_at(shiftp,t) for t in [1.,T,T5]]}}
    boundaries={}
    for c in config()['corruptions']:
        gains=[]
        for l in range(6):
            rr=[x for x in rows if x['seed']==1 and x['corruption']==c and x['level']==l];g=rr[0]['ece']['value']-rr[1]['ece']['value'];gains.append(g)
        boundaries[c]={'gains':gains,'minimum_level_g_nonpositive':next((i for i,g in enumerate(gains) if g<=0),None)}
    write_json(R/'E2.json',{'rows':rows,'pairs':pairs,'statistics':pair_statistics([p['delta'] for p in pairs]),'T5':diagnostic,'boundaries':boundaries,'analysis_units':'three genuine training seeds','preregistration_commit':evidence['commit']})

def e3():
    eps=config()['profile']['lab8_pgd_eps_255']/255;rows=[]
    for arch in ['cnn','vit']:
        sk=arch+'_1';T=read(R/'temperatures.json')[sk]['temperature']
        for e in [0,1,2,4,8]:
            p=evaluate(f'FGSM_{sk}_{e}',sk,attack_cfg={'name':'fgsm','epsilon':e/255})
            rows.append({'source_key':sk,'attack':'FGSM','eps_255':e,'RUN_ID':p.name,'K':1,'R':1,'metrics':[metrics_at(p,t) for t in [1.,T]]})
            calibrated_record(p,T)
        for K,Rr in [(10,1),(20,2)]:
            p=evaluate(f'PGD_{sk}_{K}_{Rr}',sk,attack_cfg={'name':'pgd','epsilon':eps,'K':K,'R':Rr})
            rows.append({'source_key':sk,'attack':'PGD','eps_255':eps*255,'K':K,'R':Rr,'alpha':2.5*eps/K,'RUN_ID':p.name,'metrics':[metrics_at(p,t) for t in [1.,T]]});calibrated_record(p,T)
        p=evaluate('random_'+sk,sk,attack_cfg={'name':'random','epsilon':eps})
        rows.append({'source_key':sk,'attack':'random','eps_255':eps*255,'RUN_ID':p.name,'metrics':[metrics_at(p,t) for t in [1.,T]]})
        write_json(R/'E3_progress.json',{'rows':rows})
    write_json(R/'E3.json',{'rows':rows,'N':10000,'shared_test_IDs_hash':digest(role_data('test')[2])})

def selective_ood():
    rows=[]
    for sk,Tinfo in read(R/'temperatures.json').items():
        if sk not in ['cnn_1','cnn_2','cnn_3','vit_1']:continue
        tp=Path(registry()['clean_test_'+sk]);op=evaluate('OOD_'+sk,sk,'ood');td=outputs(tp);od=outputs(op)
        T=Tinfo['temperature'];calibrated_record(op,T)
        model,source=source_model(sk);p,m=observation('selective_ood_'+sk,{'ID_RUN_ID':tp.name,'OOD_RUN_ID':op.name,'fit_RUN_ID':Tinfo['RUN_ID'],'OOD_semantic_labels_used':False},model,source)
        result={'source_key':sk,'RUN_ID':p.name,'ID_RUN_ID':tp.name,'OOD_RUN_ID':op.name,'temperature_fit_RUN_ID':Tinfo['RUN_ID'],'conditions':[]}
        for temp in [1.,T]:
            iu=uncertainty(td['logits'],temp);ou=uncertainty(od['logits'],temp);correct=(td['logits'].argmax(1)==td['labels']).numpy();conditions={}
            for measure in iu:
                u=iu[measure].numpy();o=ou[measure].numpy();flags=np.r_[np.zeros(len(u),dtype=bool),np.ones(len(o),dtype=bool)]
                conditions[measure]={'selective':risk_coverage(u,correct),'ood':detection_metrics(np.r_[u,o],flags),'ID_histogram':np.histogram(u,bins=40)[0].tolist(),'OOD_confidence_gt_0.9':float((ou['msp']<.1).double().mean())}
            result['conditions'].append({'temperature':temp,'measures':conditions})
        finish(p,m,result);rows.append(result)
    write_json(R/'selective_ood.json',rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['E1','E2','E3','selective']);args=parser.parse_args()
    {'E1':e1,'E2':e2,'E3':e3,'selective':selective_ood}[args.phase]()
