"""Serial, restartable Lab5 stages. Every optimization goes through Lab0 Runner."""
import argparse,copy,json,math,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from src.graph.experiments import config,run_train,evaluate,register_graph,smoothing_report
from src.graph.datasets import load_graph
from src.graph.stats import graph_stats,degrees
from src.graph.perturb import delete_edges,add_edges
from src.graph.null_models import degree_preserving_null,erdos_renyi
from src.diagnostics.oversmoothing import select_depth,classify_depth
from src.utils.io import write_json,file_hash

R=Path('experiments/lab5')
INDEX=R/'run_index.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def index():return read(INDEX) if INDEX.exists() else {'training':{},'evaluations':{},'smoothing':{}}
def train(key,cfg):
    idx=index()
    if key not in idx['training']:
        p=run_train(cfg);s=read(Path(p)/'summary.json')
        if s['status']!='COMPLETED':raise RuntimeError('Training paused/failed: '+p)
        idx['training'][key]=p;write_json(INDEX,idx)
    return idx['training'][key]
def ev(key,source,graph=None):
    idx=index()
    if key not in idx['evaluations']:
        idx['evaluations'][key]=evaluate(source,graph);write_json(INDEX,idx)
    return idx['evaluations'][key]
def graph(name,edges,params):
    p=R/'graphs'/f'{name}.json'
    return read(p) if p.exists() else register_graph(name,edges,params)
def selected(kind):return read(R/'E1_selection_lock.json')['selected'][kind]
def tuned(kind='gcn',**kwargs):
    s=selected(kind);return config(kind=kind,lr=s['lr'],wd=s['wd'],**kwargs)

def e1():
    candidates={}
    for kind in ['mlp','gcn','gat']:
        rows=[]
        for lr in [.01,.005]:
            for wd in [.0005,.005]:
                p=train(f'E1_{kind}_{lr}_{wd}',config(kind=kind,lr=lr,wd=wd,role='E1_search'))
                s=read(Path(p)/'summary.json');rows.append({'kind':kind,'lr':lr,'wd':wd,'source':p,'best_val':s['best_accuracy'],'best_epoch':s['best_epoch']})
        candidates[kind]=rows
    lock=R/'E1_selection_lock.json'
    if not lock.exists():
        write_json(lock,{'locked_at':datetime.now(timezone.utc).isoformat(),'criterion':'max val, earliest epoch; ties grid order',
           'test_read_for_selection':False,'candidates':candidates,'selected':{k:max(v,key=lambda r:r['best_val']) for k,v in candidates.items()}})
    ref=train('reference',config(width=16,role='fixed_reference'))
    base={k:ev('E1_'+k,selected(k)['source']) for k in candidates};reference=ev('reference',ref)
    g=load_graph('data/lab5/original.pt');a=g['edge_index'];n=len(g['y']);d=degrees(a,n).double();rows=[]
    for p in [0,.1,.3,.5,.8,1]:
        name='original' if p==0 else f'delete_{p}'
        if p:
            graph(name,delete_edges(a,p,1),{'operation':'delete_edges','p':p,'seed':1,'unit':'one undirected edge; both directed records together','common_random_numbers_across_p':True})
        gp=load_graph(f'data/lab5/{name}.pt');st=graph_stats(gp['edge_index'],n,gp['y']);dp=degrees(gp['edge_index'],n)
        A=base['gcn'] if p==0 else ev(f'E1_A_{p}',selected('gcn')['source'],name)
        B=base['gcn'] if p==0 else ev(f'E1_B_{p}',train(f'E1_B_{p}',tuned(graph=name,role='E1_deletion')))
        expected=float((p**d).sum());dev=abs(st['isolated_nodes']-expected)/expected if expected else 0.
        rows.append({'p':p,'graph':name,'A_test':A['test'],'B_test':B['test'],'A_RUN_ID':A['RUN_ID'],'B_RUN_ID':B['RUN_ID'],
             'isolated':st['isolated_nodes'],'expected_isolated':expected,'relative_deviation':dev,
             'isolation_acceptance':'PASS' if p not in [.3,.5,.8] or dev<=.2 else 'FAIL',
             'isolated_test':int(((dp==0)&g['test_mask']).sum()),'remaining_edges':st['undirected_edges'],'removed_edges':5278-st['undirected_edges'],
             'hE':st['hE'],'graph_sha256':st['graph_sha256']})
    mlp=base['mlp']['test']
    result={'models':base,'reference':reference,'reference_acceptance':'PASS' if abs(reference['test']-.815)<=.03 else 'INVESTIGATE',
            'Gtop':{k:base[k]['test']-mlp for k in ['gcn','gat']},'deletion':rows,
            'B_first_at_or_below_MLP':next((r for r in rows if r['B_test']<=mlp),None)}
    write_json(R/'E1.json',result)
    # This prediction exists before added graph generation, and is committed with E2 prereg.
    pred=R/'CF2_prediction.json'
    if not pred.exists():
        st=graph_stats(a,n,g['y']);E=st['undirected_edges'];added=math.floor(.5*E)
        write_json(pred,{'recorded_at':datetime.now(timezone.utc).isoformat(),'hE':st['hE'],'h0':st['h0'],'E':E,'added':added,
           'predicted_hE':(E*st['hE']+added*st['h0'])/(E+added),'seed':1,'formula':'(E*hE+floor(.5E)*h0)/(E+floor(.5E)); class-frequency approximation',
           'added_graph_not_yet_generated':not Path('data/lab5/add_0.5.pt').exists()})
    print('E1 COMPLETE',json.dumps({'test':{k:v['test'] for k,v in base.items()},'Gtop':result['Gtop'],'isolation':[r['isolation_acceptance'] for r in rows]}),flush=True)

def prereg_gate():
    p=R/'preregistration_evidence.json'
    if not p.exists():raise RuntimeError('E2 requires actual Git preregistration evidence')
    e=read(p);commit=e['commit']
    subprocess.run(['git','-c','safe.directory=D:/神经网络','merge-base','--is-ancestor',commit,'HEAD'],check=True)
    content=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show',commit+':experiments/lab5/hypothesis.md'])
    import hashlib
    if hashlib.sha256(content).hexdigest()!=file_hash(R/'hypothesis.md'):raise RuntimeError('Preregistered hypothesis changed')

def e2():
    prereg_gate();g=load_graph('data/lab5/original.pt');a=g['edge_index'];n=len(g['y']);rows=[]
    for seed in [1,2,3]:
        name=f'null_{seed}'
        if not (R/'graphs'/f'{name}.json').exists():
            edge,stats=degree_preserving_null(a,10,seed,n);graph(name,edge,{'operation':'degree_preserving_null',**stats})
        original=ev(f'E2_original_{seed}',train(f'E2_original_{seed}',tuned(seed=seed,role='E2_original')))
        null=ev(f'E2_null_{seed}',train(f'E2_null_{seed}',tuned(seed=seed,graph=name,role='E2_null')))
        mlp=ev(f'E2_mlp_{seed}',train(f'E2_mlp_{seed}',tuned(kind='mlp',seed=seed,role='E2_MLP')))
        st=read(R/'graphs'/f'{name}.json');r=st['statistics'];swap=st['parameters']
        rows.append({'seed':seed,'original':original,'null':null,'mlp':mlp,'delta':original['test']-null['test'],
            'null_hE':r['hE'],'hdeg':r['hdeg'],'null_components':r['components'],'swap_success':swap['success_rate'],
            'degree_exact':swap['degree_sequence_exactly_preserved'],'hE_relative_gap':abs(r['hE']-r['hdeg'])/r['hdeg']})
    graph('ER',erdos_renyi(n,5278,1),{'operation':'erdos_renyi','nodes':n,'undirected_edges':5278,'seed':1,'degree_preserved':False})
    er=ev('E2_ER',train('E2_ER',tuned(graph='ER',role='E2_ER')))
    deltas=np.array([r['delta'] for r in rows]);mean=float(deltas.mean());sd=float(deltas.std(ddof=1));same=bool((deltas>0).all() or (deltas<0).all())
    sm=float(np.std([r['original']['val'] for r in rows],ddof=1))
    result={'paired':rows,'ER':er,'mean_delta':mean,'sample_sd_delta':sd,'same_nonzero_sign':same,
       'teacher_conclusion':'specific edge structure difference established' if same and abs(mean)>=2*sd else 'difference not established',
       'sm_validation':sm,'three_seeds_scope':'one fixed public split only; not variability over splits',
       'seed1_disclosure':'E2 seed1 originalGCN and MLP are fresh formal repetitions after prereg, but same seed/config already explored in E1; no claim of unseen seed1 data'}
    write_json(R/'E2.json',result);print('E2 COMPLETE',json.dumps({'mean':mean,'sd':sd,'sm':sm}),flush=True)

def e3_train():
    # No E3 test evaluation until every formal training condition is complete.
    idx=index();idx['training']['E3_plain_2']=idx['training']['E2_original_1'];write_json(INDEX,idx)
    for depth in [3,6,12,24]:train(f'E3_plain_{depth}',tuned(depth=depth,role='E3_plain'))
    train('E3_residual_24',tuned(depth=24,residual=True,role='E3_residual'))
    train('CF1_GAT_null',tuned(kind='gat',graph='null_1',role='CF1'))
    pred=read(R/'CF2_prediction.json');assert pred['added_graph_not_yet_generated']
    graph('add_0.5',add_edges(load_graph('data/lab5/original.pt')['edge_index'],.5,1,2708),{'operation':'add_edges','p':.5,'seed':1,'added_edges':2639,'prediction_file':'experiments/lab5/CF2_prediction.json'})
    train('CF2_added',tuned(graph='add_0.5',role='CF2'))
    idx=index();rows=[]
    for depth in [2,3,6,12,24]:
        source=idx['training'][f'E3_plain_{depth}'];s=read(Path(source)/'summary.json')
        rows.append({'depth':depth,'source':source,'val':s['best_accuracy'],'best_epoch':s['best_epoch']})
    lock=R/'all_training_selection_lock.json'
    if not lock.exists():write_json(lock,{'locked_at':datetime.now(timezone.utc).isoformat(),'E3_candidates_val_only':rows,
        'lstar_validation':select_depth(rows),'training_index':idx['training'],
        'test_based_training_after_this_lock_forbidden':True,'E1_depth2_test_preexposure_disclosed':True})
    print('ALL MODEL TRAINING AND VAL SELECTION LOCKED',flush=True)

def analysis():
    lock=read(R/'all_training_selection_lock.json');idx=index();plain=[]
    for depth in [2,3,6,12,24]:plain.append({'depth':depth,**ev(f'E3_plain_{depth}',idx['training'][f'E3_plain_{depth}'])})
    residual=ev('E3_residual',idx['training']['E3_residual_24']);cf1=ev('CF1_GAT',idx['training']['CF1_GAT_null']);cf2=ev('CF2_added',idx['training']['CF2_added'])
    for key,source in [('untrained',None)]+[(f'plain_{d}',idx['training'][f'E3_plain_{d}']) for d in [2,3,6,12,24]]+[('residual_24',idx['training']['E3_residual_24'])]:
        current=index()
        if key not in current['smoothing']:
            s=smoothing_report(source,untrained=source is None);current['smoothing'][key]=s;write_json(INDEX,current)
            print('SMOOTHING COMPLETE',key,flush=True)
    idx=index();sm=read(R/'E2.json')['sm_validation'];best=next(r for r in plain if r['depth']==lock['lstar_validation'])
    bestlayer=idx['smoothing'][f"plain_{best['depth']}"]['layers'][-1]
    for row in plain:
        last=idx['smoothing'][f"plain_{row['depth']}"]['layers'][-1];row.update(S=last['S'],energy=last['energy'],probe_val=last['probe_val'],probe_RUN_ID=last['probe_RUN_ID'])
        row['GI']=row['gradient_diagnostic']['GI'];row['val_drop']=best['val']-row['val'];row['significant_drop']=row['val_drop']>2*sm
        source=Path(idx['training'][f"E3_plain_{row['depth']}"])
        summary=read(source/'summary.json');row['last_training_rhoT']=summary['train_loss']/math.log(7)
        gradients=[json.loads(v) for v in (source/'gradients.jsonl').read_text().splitlines()]
        medians={k:float(np.median([r['weight_g'][k] for r in gradients])) for k in gradients[0]['weight_g']}
        row['training_gradient_medians']=medians;values=list(medians.values())
        row['GI_training_median']=math.log10((max(values)+1e-12)/(min(values)+1e-12))
        row['diagnosis']=classify_depth(row['rhoT'],row['GI'],row['energy'],row['probe_val'],bestlayer['energy'],bestlayer['probe_val']) if row['significant_drop'] else 'no significant decline'
    layer=idx['smoothing']['residual_24']['layers'][-1];residual.update(S=layer['S'],energy=layer['energy'],probe_val=layer['probe_val'],GI=residual['gradient_diagnostic']['GI'])
    result={'plain':plain,'residual':residual,'lstar_validation':lock['lstar_validation'],'sm':sm,'threshold':2*sm,
        'test_argmax_posthoc':max(plain,key=lambda r:r['test'])['depth'],'test_argmax_scope':'CF5 only; all models and validation choices locked before this operation; no new training or tuning',
        'CF1_GAT_null':cf1,'CF2_added':cf2}
    write_json(R/'E3.json',result);print('E3 ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['e1','e2','e3_train','analysis']);args=p.parse_args();globals()[args.stage]()
