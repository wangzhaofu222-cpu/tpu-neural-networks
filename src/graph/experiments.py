"""Graph operations/evaluation use the Lab0 provenance and ExperimentRunner."""
import copy,json,math,time
from pathlib import Path
import torch,yaml
from torch.nn import functional as F
from src.config import validate
from src.engine.runner import ExperimentRunner
from src.models.small_cnn import build_model
from src.graph.datasets import load_graph,datasets,fingerprints
from src.graph.stats import graph_stats,degrees
from src.utils.io import write_json,digest,file_hash
from src.utils.provenance import create_run,git_state
from src.diagnostics.oversmoothing import gradient_record,energy,cosine_similarity,sample_pairs,normalized_operator

ROOT=Path('experiments/lab5')

def config(kind='gcn',seed=1,graph='original',role='E1',depth=2,width=64,purpose='formal',lr=.01,wd=.0005,residual=False):
    cfg=yaml.safe_load(Path('configs/lab5.yaml').read_text(encoding='utf-8'))
    cfg['experiment'].update(name=role,role=role,seed=seed,purpose=purpose)
    cfg['data']['graph_path']=f'data/lab5/{graph}.pt'
    cfg['model'].update(kind=kind,depth=depth,width=width,residual=residual,graph_path=cfg['data']['graph_path'],initialization_seed=seed)
    cfg['optimizer'].update(lr=lr,weight_decay=wd)
    validate(cfg);return cfg

def run_train(cfg):
    validate(cfg);return str(ExperimentRunner(cfg).run()).replace('\\','/')

def selected_model(source,which='best'):
    p=Path(source);cfg=yaml.safe_load((p/'config.resolved.yaml').read_text(encoding='utf-8'))
    model=build_model(cfg['model']).to(cfg['runtime']['device'])
    state=torch.load(p/f'checkpoints/{which}.pt',weights_only=False,map_location=cfg['runtime']['device'])
    model.load_state_dict(state['model']);model.eval();return model,cfg,state

def operation(cfg,info,model,kind,parent=None):
    code=git_state()
    if cfg['experiment']['purpose']=='formal' and code['dirty']:raise RuntimeError('Formal graph operation needs committed source/config')
    path,manifest=create_run(cfg,code,info,model,parent)
    manifest.update(status='COMPLETED_'+kind,global_step=0,completed_epoch=0)
    write_json(path/'manifest.json',manifest)
    return path,manifest

def register_graph(name,edge_index,parameters):
    g=load_graph('data/lab5/original.pt');before=fingerprints(g);g['edge_index']=edge_index
    assert fingerprints(g)==before
    dest=Path(f'data/lab5/{name}.pt');torch.save(g,dest)
    cfg=config(graph=name,role='graph_generation');cfg['perturbation']=parameters
    _,info=datasets(cfg);model=build_model(cfg['model'])
    path,manifest=operation(cfg,info,model,'GRAPH_GENERATION')
    result={'RUN_ID':path.name,'graph':name,'parameters':parameters,'statistics':info['graph_statistics'],
            'graph_path':str(dest),'graph_file_sha256':file_hash(dest),'node_data_unchanged':True}
    write_json(path/'summary.json',result);write_json(ROOT/'graphs'/f'{name}.json',result)
    return result

def evaluate(source,graph=None,role='evaluation'):
    model,cfg,state=selected_model(source)
    if graph:cfg['data']['graph_path']=f'data/lab5/{graph}.pt';cfg['model']['graph_path']=cfg['data']['graph_path']
    cfg['experiment']['role']=role
    g=load_graph(cfg['data']['graph_path']);_,info=datasets(cfg)
    path,manifest=operation(cfg,info,model,'EVALUATION',Path(source).name)
    dev=next(model.parameters()).device;x=g['x'].to(dev);edges=g['edge_index'].to(dev)
    with torch.no_grad():logits=model(x,edges);pred=logits.argmax(1).cpu()
    result={'RUN_ID':path.name,'source_RUN_ID':Path(source).name,'source_checkpoint_sha256':file_hash(Path(source)/'checkpoints/best.pt'),
        'best_epoch':state['epoch'],'graph_statistics':info['graph_statistics'],'graph':graph or Path(cfg['data']['graph_path']).stem}
    for r in ['train','val','test']:
        m=g[r+'_mask'];result[r]=float((pred[m]==g['y'][m]).float().mean())
        result[r+'_loss']=F.cross_entropy(logits.cpu()[m],g['y'][m]).item()
    result['rhoT']=result['train_loss']/math.log(cfg['model']['classes'])
    model.zero_grad(set_to_none=True);mask=g['train_mask'].to(dev)
    F.cross_entropy(model(x,edges)[mask],g['y'].to(dev)[mask]).backward()
    result['gradient_diagnostic']=gradient_record(model)
    torch.save({'prediction':pred,'logits':logits.cpu(),'roles':{r:g[r+'_mask'] for r in ['train','val','test']}},path/'predictions.pt')
    write_json(path/'predictions.json',{'prediction':pred.tolist(),'roles':{r:torch.where(g[r+'_mask'])[0].tolist() for r in ['train','val','test']}})
    write_json(path/'summary.json',result);return result

def smoothing_report(source=None,untrained=False):
    """Save all layer features and use the SAME harness for every frozen linear probe."""
    g=load_graph('data/lab5/original.pt');pairs=sample_pairs(len(g['y']));edge=g['edge_index']
    if untrained:
        cfg=config(role='untrained_propagation');cfg['model']['kind']='gcn';model=build_model(cfg['model'])
        op=normalized_operator(edge,len(g['y']));h=g['x'].double();features=[h.float()]
        for _ in range(24):h=torch.sparse.mm(op,h);features.append(h.float())
    else:
        model,cfg,_=selected_model(source);dev=next(model.parameters()).device
        with torch.no_grad():features=[g['x']]+[v.detach().cpu() for v in model.forward_features(g['x'].to(dev),edge.to(dev))]
        cfg=copy.deepcopy(cfg);cfg['experiment']['role']='smoothing_analysis'
    _,info=datasets(cfg);path,manifest=operation(cfg,info,model,'SMOOTHING',Path(source).name if source else None)
    cache=Path('data/lab5/features')/path.name;cache.mkdir(parents=True)
    records=[]
    for layer,h in enumerate(features):
        f=cache/f'layer{layer}.pt';torch.save(h,f)
        probe=copy.deepcopy(cfg);probe['experiment'].update(name='layer_probe',role='layer_probe',seed=1)
        probe['data'].update(name='graph_features',graph_path='data/lab5/original.pt',features_path=str(f))
        probe['model']={'name':'linear_probe','input_dim':h.shape[1],'classes':7}
        probe['optimizer'].update(lr=.01,weight_decay=.001)
        probe['train']['epochs']=200
        probe['runtime']['device']='cpu'
        p=Path(run_train(probe));s=json.loads((p/'summary.json').read_text())
        records.append({'layer':layer,'S':cosine_similarity(h,pairs),'energy':energy(h,edge),
          'probe_val':s['best_accuracy'],'probe_best_epoch':s['best_epoch'],'probe_RUN_ID':p.name,
          'feature_path':str(f),'feature_sha256':file_hash(f)})
    result={'RUN_ID':path.name,'source_RUN_ID':Path(source).name if source else None,'untrained':untrained,
        'pair_count':len(pairs),'pair_seed':1729,'pair_hash':digest(pairs.tolist()),
        'zero_vector_convention':'cosine contribution zero if either vector zero; zero full H energy null',
        'probe_protocol':'frozen features; column mean/std fitted on train only; 200 full-batch Adam epochs lr.01 wd.001; val best earliest; no test evaluation',
        'operator':'Dtilde^-1/2(A+I)Dtilde^-1/2; energy uses undirected raw edge sum after degree correction',
        'layers':records}
    write_json(path/'summary.json',result)
    if source:
        sp=Path(source)/'summary.json';s=json.loads(sp.read_text());s['smoothing']=result;write_json(sp,s)
    return result
