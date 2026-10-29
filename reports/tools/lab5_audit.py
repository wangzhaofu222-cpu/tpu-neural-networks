"""Independent recalculation, provenance chronology and historical protection audit."""
import csv,json,math,subprocess,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch,yaml
from src.utils.io import file_hash,write_json,digest
from src.graph.datasets import load_graph,fingerprints
from src.graph.stats import graph_stats,degrees,assert_simple
from src.graph.perturb import delete_edges,add_edges
from src.graph.null_models import degree_preserving_null,erdos_renyi
from src.models.small_cnn import build_model
from src.diagnostics.oversmoothing import energy,cosine_similarity,sample_pairs

R=Path('experiments/lab5')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def git(*args):return subprocess.check_output(['git','-c','safe.directory=D:/神经网络',*args]).decode().strip()
def main():
    checks=[]
    def check(name,ok,detail=None):
        checks.append({'check':name,'status':'PASS' if bool(ok) else 'FAIL','detail':detail})
    historical=read('experiments/lab4_verification/historical_files.json')
    changed=[p for p,h in historical.items() if not Path(p).exists() or file_hash(p)!=h]
    check('historical_Lab0_to_4_bytes_unchanged',not changed,changed)
    check('Lab4_Q2_still_BLOCKED',read('experiments/lab4_verification/audit.json')['Q2']=='BLOCKED')
    git('merge-base','--is-ancestor','709e39cf7d7bc8ab4759c043c1795e063356d292','HEAD');check('Lab4_final_commit_ancestor',True)
    idx=read(R/'run_index.json');e1=read(R/'E1.json');e2=read(R/'E2.json');e3=read(R/'E3.json')
    g=load_graph('data/lab5/original.pt');n=len(g['y']);a=g['edge_index'];base_f=fingerprints(g);degree=degrees(a,n)
    graph_evidence=[]
    for p in sorted((R/'graphs').glob('*.json')):
        r=read(p);params=r['parameters'];name=r['graph'];x=load_graph(r['graph_path']);op=params['operation']
        if op=='delete_edges':edges=delete_edges(a,params['p'],params['seed'])
        elif op=='degree_preserving_null':edges,meta=degree_preserving_null(a,params['Q'],params['seed'],n)
        elif op=='erdos_renyi':edges=erdos_renyi(n,params['undirected_edges'],params['seed'])
        elif op=='add_edges':edges=add_edges(a,params['p'],params['seed'],n)
        else:raise ValueError(op)
        assert_simple(x['edge_index'],n);st=graph_stats(edges,n,g['y'])
        ok=torch.equal(edges,x['edge_index']) and st==r['statistics'] and fingerprints(x)==base_f
        check('regenerate_graph_'+name,ok)
        graph_evidence.append({'graph':name,'RUN_ID':r['RUN_ID'],'recomputed':ok,'sha256':st['graph_sha256']})
    checkpoint_inventory=[];unique_train=set(idx['training'].values())
    probe_sources={str(Path('runs/lab5')/l['probe_RUN_ID']) for s in idx['smoothing'].values() for l in s['layers']}
    all_train=unique_train|probe_sources
    prereg=read(R/'preregistration_evidence.json');prereg_time=datetime.fromisoformat(prereg['committed_at'])
    hypothesis_blob=subprocess.check_output(['git','-c','safe.directory=D:/神经网络','show',prereg['commit']+':experiments/lab5/hypothesis.md'])
    import hashlib
    check('E2_hypothesis_immutable_git_blob',hashlib.sha256(hypothesis_blob).hexdigest()==file_hash(R/'hypothesis.md'))
    git('merge-base','--is-ancestor',prereg['commit'],'HEAD')
    train_seconds=0.;formal_ids=[]
    for name in sorted(all_train):
        p=Path(name);m=read(p/'manifest.json');s=read(p/'summary.json');cfg=yaml.safe_load((p/'config.resolved.yaml').read_text(encoding='utf-8'))
        metrics=[json.loads(v) for v in (p/'metrics.jsonl').read_text().splitlines()]
        check('full200_'+p.name,m['status']=='COMPLETED' and m['completed_epoch']==200 and m['global_step']==200 and len(metrics)==200 and not m['git_dirty'])
        best=max(metrics,key=lambda r:r['validation_metric']);check('val_epoch_selection_'+p.name,best['epoch']==s['best_epoch'] and best['validation_metric']==s['best_accuracy'])
        for f in sorted((p/'checkpoints').glob('*.pt')):
            state=torch.load(f,weights_only=False,map_location='cpu')
            required={'model','optimizer','scheduler','scaler','epoch','global_step','config_hash','source_hash'}
            check('checkpoint_'+p.name+'_'+f.stem,required<=set(state))
            checkpoint_inventory.append({'path':str(f),'sha256':file_hash(f),'epoch':state['epoch'],'RUN_ID':p.name})
        if cfg['experiment']['role'] in {'E2_original','E2_null','E2_MLP'}:
            check('E2_prereg_before_'+p.name,datetime.fromisoformat(m['started_at'])>prereg_time)
        formal_ids.append(p.name);train_seconds+=s['training_seconds']
    # Recompute each saved evaluation using the actual best checkpoint and condition graph.
    evaluation_inventory=[]
    for key,r in idx['evaluations'].items():
        p=Path('runs/lab5')/r['RUN_ID'];cfg=yaml.safe_load((p/'config.resolved.yaml').read_text(encoding='utf-8'))
        gg=load_graph(cfg['data']['graph_path']);model=build_model(cfg['model']).cpu().eval()
        source=Path('runs/lab5')/r['source_RUN_ID'];weights=torch.load(source/'checkpoints/best.pt',weights_only=False,map_location='cpu')
        model.load_state_dict(weights['model'])
        with torch.no_grad():logits=model(gg['x'],gg['edge_index']);pred=logits.argmax(1)
        saved=read(p/'predictions.json');ok=pred.tolist()==saved['prediction']
        for role in ['train','val','test']:
            mask=gg[role+'_mask'];acc=float((pred[mask]==gg['y'][mask]).float().mean());ok=ok and acc==r[role]
        loss=torch.nn.functional.cross_entropy(logits[gg['train_mask']],gg['y'][gg['train_mask']]).item()
        check('evaluate_recomputed_'+key,ok and abs(loss/math.log(7)-r['rhoT'])<1e-8)
        evaluation_inventory.append({'key':key,'RUN_ID':r['RUN_ID'],'prediction_exact':ok})
    # Independently recalculate all S/energy and verify probe references/summary mirrors.
    pairs=sample_pairs(n);layer_count=0
    for name,s in idx['smoothing'].items():
        ok=True
        for row in s['layers']:
            h=torch.load(row['feature_path'],weights_only=False);ee=energy(h,a);ss=cosine_similarity(h,pairs)
            probe=read(Path('runs/lab5')/row['probe_RUN_ID']/'summary.json')
            ok=ok and ee==row['energy'] and ss==row['S'] and probe['best_accuracy']==row['probe_val']
            layer_count+=1
        check('smoothing_recomputed_'+name,ok and s['pair_hash']==digest(pairs.tolist()))
    d=np.array([r['original']['test']-r['null']['test'] for r in e2['paired']]);sm=np.std([r['original']['val'] for r in e2['paired']],ddof=1)
    check('E2_independent_paired_statistics',abs(float(d.mean())-e2['mean_delta'])<1e-12 and abs(float(d.std(ddof=1))-e2['sample_sd_delta'])<1e-12)
    check('E2_degree_null_acceptance',all(r['degree_exact'] and r['swap_success']>=.5 and r['hE_relative_gap']<=.2 for r in e2['paired']))
    check('E1_isolation_threshold',all(r['relative_deviation']<=.2 for r in e1['deletion'] if r['p'] in [.3,.5,.8]))
    check('fixed_reference_within3pp',abs(e1['reference']['test']-.815)<=.03)
    check('E3_val_depth_and_threshold',max(e3['plain'],key=lambda r:r['val'])['depth']==e3['lstar_validation'] and abs(float(sm)*2-e3['threshold'])<1e-12)
    check('all_assigned_depths',set(r['depth'] for r in e3['plain'])=={2,3,6,12,24})
    check('CF4_predeclared_witness',read(R/'CF4_witnesses.json')['status']=='PASS')
    lock=read(R/'all_training_selection_lock.json');locktime=datetime.fromisoformat(lock['locked_at'])
    model_times=[datetime.fromisoformat(read(Path(p)/'manifest.json')['started_at']) for p in unique_train]
    check('no_model_training_after_selection_lock',all(t<locktime for t in model_times))
    # Test only CF5 inspections were after lock; baseline depth2 existed in E1, disclosed.
    check('E3_tests_after_lock',all(datetime.fromisoformat(read(Path('runs/lab5')/idx['evaluations'][f'E3_plain_{d}']['RUN_ID']/'manifest.json')['started_at'])>locktime for d in [2,3,6,12,24]))
    tests=ET.parse('docs/environment/tests-lab5-final.xml').getroot().find('testsuite')
    check('final_tests',int(tests.attrib['failures'])==0 and int(tests.attrib['errors'])==0)
    write_json(R/'checkpoint_inventory.json',checkpoint_inventory);write_json(R/'graph_regeneration_audit.json',graph_evidence)
    write_json(R/'evaluation_recalculation.json',evaluation_inventory)
    stats={'model_training_runs':len(unique_train),'linear_probe_training_runs':len(probe_sources),'formal_training_runs':len(all_train),
       'formal_training_seconds':train_seconds,'formal_optimizer_steps':200*len(all_train),'formal_checkpoints':len(checkpoint_inventory),
       'layer_summaries':layer_count,'final_tests_passed':int(tests.attrib['tests']),
       'budget_runs':6,'budget_training_seconds':sum(m['training_seconds'] for m in read(R/'budget.json')['measurements'] if m['status']=='COMPLETED'),
       'experiment_start':min(read(Path(p)/'manifest.json')['started_at'] for p in unique_train),
       'audit_finished_at':datetime.now(timezone.utc).isoformat()}
    write_json(R/'execution_statistics.json',stats)
    result={'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL','checks':checks,'check_count':len(checks),
       'failed':[c for c in checks if c['status']=='FAIL'],'statistics':stats,
       'teacher_split_independently_verified':False,'public_masks_user_authorized':True,
       'Lab4_Q2':'BLOCKED (unchanged; absent course color protocol)','Lab5_BLOCKED':[],
       'scope':'no formal report generated; no Lab6 executed'}
    write_json(R/'final_audit.json',result);print(json.dumps({'status':result['status'],'failed':result['failed'],'statistics':stats}))
if __name__=='__main__':main()
