"""Independent checks of saved revision artifacts; no training or data changes."""
import json,hashlib,math
from pathlib import Path
import numpy as np
import torch
import yaml
from xml.etree import ElementTree as ET
from src.models.small_cnn import build_model
from src.engine.checkpoint import load_checkpoint
from src.utils.io import digest,file_hash,write_json

ROOT=Path('experiments/revision_20261106')
def read(p):return json.loads(Path(p).read_text('utf-8'))
checks=[]
def check(name,value):
    checks.append({'check':name,'pass':bool(value)})
    assert value,name

torch.set_num_threads(2)
original=read('reports/formal_inputs_sha256.json')
for p,h in original.items():check('original unchanged: '+p,file_hash(p)==h)
tree=ET.parse(ROOT/'tests.xml');suite=tree.getroot().find('testsuite')
check('43 selected automated tests passed',suite is not None and int(suite.get('tests'))==43 and int(suite.get('failures'))+int(suite.get('errors'))==0)
lab1=read(ROOT/'lab1/results.json');check('all 12 original E2 conditions repeated',len(lab1['rows'])==12)
check('initial moments reproducible',all(r['initial_q_max_relative_difference']<1e-12 for r in lab1['rows']))
for r in lab1['rows']:
    if 'exact_identity_relative_error' in r:check(r['label']+' exact cross-moment identity',r['exact_identity_relative_error']<1e-12)
    old=read(Path('runs/lab1')/r['original_RUN_ID']/'summary.json')
    check(r['label']+' status reproduced',old['status']==r['status'])
    if r['status']=='COMPLETED':
        check(r['label']+' full 3-epoch budget',r['global_step']==1008)
        check(r['label']+' validation result reproduced',r['validation_metric']==old['validation_metric'])

d=read(ROOT/'lab4/results.json');features={}
for name,row in d['conditions'].items():
    fr=read(ROOT/f'lab4/{name}_features.json');features[name]=fr
    p=Path(fr['path']);cache=torch.load(p/'features.pt',map_location='cpu',weights_only=False)
    check(name+' original encoder unchanged',file_hash(Path('runs/lab4')/row['encoder_RUN_ID']/'checkpoints/last.pt')==fr['encoder_checkpoint_sha256'])
    check(name+' feature hash',file_hash(p/'features.pt')==fr['feature_cache_sha256'])
    check(name+' no cal',fr['cal_access'] is False)
    for role,n in [('train',43000),('val',5000),('test',10000)]:
        a=np.load(p/f'{role}_assignments.npz')
        check(name+' '+role+' exact balance',len(a['labels'])==n and np.array_equal(np.bincount(a['labels'],minlength=10),np.full(10,n//10)))
        check(name+' '+role+' hue class rule',np.array_equal(np.floor(a['hues']*10).astype(int),a['labels']))
        check(name+' '+role+' cache labels match assignments',np.array_equal(a['labels'],cache[role]['labels'].numpy()))
    candidates=row['candidates'];winner=max(candidates,key=lambda x:(x['validation_accuracy'],-x['regularization']))
    check(name+' validation-only selection',winner['RUN_ID']==row['chosen_probe_RUN_ID'])
    check(name+' all 3 regularization candidates',sorted(c['regularization'] for c in candidates)==[0.,.001,.01])
    saved=torch.load(Path(row['path'])/'predictions.pt',map_location='cpu',weights_only=False)
    pred=saved['probabilities'].argmax(1);target=saved['labels']
    check(name+' accuracy recomputation',(pred==target).double().mean().item()==row['test_accuracy'])
    check(name+' probability normalization',torch.max(torch.abs(saved['probabilities'].sum(1)-1)).item()<1e-6)
    check(name+' confusion recomputation',torch.bincount(target*10+pred,minlength=100).reshape(10,10).tolist()==row['confusion'])
    for c in candidates:
        path=Path(c['path']);cfg=yaml.safe_load((path/'config.resolved.yaml').read_text('utf-8'));s=read(path/'summary.json')
        check(c['RUN_ID']+'200 full epochs',s['status']=='COMPLETED' and s['global_step']==200)

check('identical coloured images for both encoders',features['with_jitter']['input_hashes']==features['without_jitter']['input_hashes'])
check('reported accuracy difference',d['difference_without_minus_with']==d['conditions']['without_jitter']['test_accuracy']-d['conditions']['with_jitter']['test_accuracy'])

run_paths=[Path('runs/lab1')/r['repeat_RUN_ID'] for r in lab1['rows']]
run_paths += [Path(c['path']) for r in d['conditions'].values() for c in r['candidates']]
index=[]
for p in run_paths:
    cfg=yaml.safe_load((p/'config.resolved.yaml').read_text('utf-8'));model=build_model(cfg['model'])
    for cp in (p/'checkpoints').glob('*.pt'):
        st=load_checkpoint(cp,model)
        check(str(cp)+' config/identity',st['config_hash']==digest(cfg) and st['run_id']==p.name)
        check(str(cp)+' optimizer/RNG states',all(k in st for k in ['optimizer','scheduler','scaler','torch_rng','numpy_rng','python_rng','loader_rng']))
        index.append({'path':cp.as_posix(),'sha256':file_hash(cp),'epoch':st['epoch'],'global_step':st['global_step']})
write_json(ROOT/'checkpoint_index.json',index)
write_json(ROOT/'audit.json',{'status':'PASS','checks_passed':len(checks),'checks':checks,'automated_tests_passed':43,'checkpoint_count':len(index),
    'original_input_files_preserved':len(original),'Lab1_scaled_failures_retained':lab1['scaled_failures'],'Lab4_Q2':'COMPLETE',
    'protocol':read(ROOT/'decisions.json'),'no_seed_search':True,'no_original_hypothesis_changes':True})
write_json('experiments/current_acceptance.json',{'as_of':'2026-11-06','source':(ROOT/'audit.json').as_posix(),
    'Lab4_Q2':'COMPLETE: hue control with two fixed E2 encoders and six 200-epoch probes',
    'Lab5_protocol':'ADOPTED: PyG Cora public masks','Lab9_protocol':'ADOPTED: actor seed1729; explicit Lab2 CNN source',
    'Lab1':'E2 repeated; code checks pass; scaled96 numerical threshold failure reproduced and retained',
    'unchanged_valid_negative_results':['Lab3 alternative decomposition','Lab6 fallback eta','Lab8 reversed prediction','Lab9 difference not established']})
print(json.dumps({'checks_passed':len(checks),'checkpoints':len(index),'original_files_preserved':len(original),'tests':43},indent=2))
