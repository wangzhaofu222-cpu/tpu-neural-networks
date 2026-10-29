import copy,json,math
from pathlib import Path
import numpy as np
import pytest
import torch
from src.graph.stats import canonical_edges,assert_simple,degrees,graph_stats,undirected_pairs
from src.graph.perturb import delete_edges,add_edges
from src.graph.null_models import degree_preserving_null,erdos_renyi
from src.models.graph import GraphNetwork
from src.diagnostics.oversmoothing import energy,normalized_operator,cosine_similarity,sample_pairs
from src.diagnostics.oversmoothing import select_depth,classify_depth
from src.graph.datasets import datasets,load_graph,fingerprints
from src.graph.experiments import config
from src.engine.runner import ExperimentRunner
from src.metrics.objective import correct_count,objective

def ring(n=10):return canonical_edges(torch.tensor([[i,(i+1)%n] for i in range(n)]).T,n)

def model(kind='gcn',depth=2):
    return GraphNetwork({'kind':kind,'input_dim':5,'width':8 if kind!='gat' else 64,'classes':3,'depth':depth,'dropout':.5,'initialization_seed':1})

def test_mandatory1_mlp_edge_independent():
    m=model('mlp').eval();x=torch.randn(10,5)
    a=m(x,ring());b=m(x,ring()[:,:0]);c=m(x,erdos_renyi(10,20,5))
    assert torch.equal(a,b) and torch.equal(a,c)

@pytest.mark.parametrize('depth',[2,3,6])
def test_mandatory2_empty_gcn_matches_mlp(depth):
    g=model('gcn',depth).eval();m=model('mlp',depth).eval()
    for a,b in zip(g.layers,m.layers):b.weight.data.copy_(a.lin.weight.data);b.bias.data.copy_(a.bias.data)
    x=torch.randn(10,5);empty=torch.empty((2,0),dtype=torch.long)
    assert (g(x,empty)-m(x,empty)).abs().max()<1e-6

@pytest.mark.parametrize('p',[0,.1,.3,.5,.8,1])
def test_mandatory3_deletion(p):
    a=ring();out=delete_edges(a,p,7);assert_simple(out,10)
    assert torch.equal(out,delete_edges(a,p,7))
    if p==0:assert torch.equal(a,out)
    if p==1:assert out.numel()==0

def test_mandatory3_addition_and_node_immutability():
    g=load_graph('data/lab5/original.pt');f=fingerprints(g);a=g['edge_index'].clone()
    new=add_edges(a,.5,1,len(g['y']));assert_simple(new,len(g['y']))
    assert new.shape[1]==a.shape[1]+2*math.floor(.5*a.shape[1]/2)
    assert fingerprints(g)==f and torch.equal(g['edge_index'],a)
    assert set(map(tuple,undirected_pairs(a).tolist()))<=set(map(tuple,undirected_pairs(new).tolist()))

def test_mandatory4_actual_cora_null():
    g=load_graph('data/lab5/original.pt');a=g['edge_index'];n=len(g['y'])
    b,r=degree_preserving_null(a,10,1,n);assert_simple(b,n)
    assert torch.equal(degrees(a,n),degrees(b,n))
    assert r['attempts']==10*a.shape[1]//2 and r['success_rate']>=.5
    st=graph_stats(b,n,g['y']);assert abs(st['hE']-st['hdeg'])/st['hdeg']<=.2

def test_mandatory5_normalized_limit_and_energy():
    a=canonical_edges(torch.cat([ring(),torch.tensor([[0,0,0],[2,3,4]])],1),10)
    op=normalized_operator(a,10);x=torch.randn(10,5,dtype=torch.double);h=x.clone()
    for _ in range(200):h=torch.sparse.mm(op,h)
    d=(degrees(a,10)+1).double();v=d.sqrt();limit=v[:,None]*(v@x/v.square().sum())[None,:]
    assert (h-limit).abs().max()<1e-4
    assert ((h/v[:,None])-(h/v[:,None])[0]).abs().max()<1e-4
    assert energy(limit,a)<1e-8
    assert energy(x,a)==pytest.approx(energy(x*1e-100,a),abs=1e-12)

def test_energy_dense_trace_and_monotonic():
    a=ring();h=torch.randn(10,5,dtype=torch.double);op=normalized_operator(a,10)
    expected=torch.trace(h.T@(torch.eye(10)-op.to_dense())@h)/h.square().sum()
    assert energy(h,a)==pytest.approx(expected.item())
    vals=[]
    for _ in range(30):vals.append(energy(h,a));h=torch.sparse.mm(op,h)
    assert all(b<=a+1e-12 for a,b in zip(vals,vals[1:]))
    assert energy(torch.zeros_like(h),a) is None

def test_gat_attention_normalization():
    m=model('gat').eval();x=torch.randn(10,5);out,(idx,alpha)=m.layers[0](x,ring(),return_attention_weights=True)
    assert out.shape==(10,64)
    sums=torch.zeros(10,8);sums.index_add_(0,idx[1],alpha)
    assert torch.allclose(sums,torch.ones_like(sums),atol=1e-6)

def test_role_masks_and_training_loss():
    cfg=config(purpose='test');ds,info=datasets(cfg);x,y=ds['train'];g=load_graph(cfg['data']['graph_path'])
    assert (y!=-100).sum()==140 and (ds['val'][1]!=-100).sum()==500
    assert set(ds)=={'train','val'} and info['split_sizes']['cal']==0
    logits=torch.randn(2708,7);assert objective(logits,y)==objective(logits[g['train_mask']],g['y'][g['train_mask']])
    assert correct_count(logits,y)[1]==140

def test_actual_cora_statistics_and_er():
    g=load_graph('data/lab5/original.pt');s=graph_stats(g['edge_index'],2708,g['y'])
    assert (s['nodes'],s['undirected_edges'],s['components'],s['isolated_nodes'])==(2708,5278,78,0)
    assert round(s['hE'],2)==.81
    a=erdos_renyi(2708,5278,1);assert_simple(a,2708);assert a.shape[1]==10556
    assert not torch.equal(degrees(a,2708),degrees(g['edge_index'],2708))
    assert graph_stats(a[:,:0],2708,g['y'])['hE'] is None

def test_pair_sampling_and_cosine():
    p=sample_pairs(2708);assert len(p)==20000 and len(set(map(tuple,p.tolist())))==20000
    assert (p[:,0]<p[:,1]).all() and torch.equal(p,sample_pairs(2708))
    x=torch.zeros(3,2);assert cosine_similarity(x,sample_pairs(3))==0
    x[:]=1;assert cosine_similarity(x,sample_pairs(3))==pytest.approx(1)

def test_local_perturbation_rng():
    a=ring();np.random.seed(12);before=np.random.get_state();torch.manual_seed(12);t=torch.get_rng_state()
    delete_edges(a,.5,1);add_edges(a,.5,1,10);degree_preserving_null(a,10,1,10);erdos_renyi(10,10,1)
    after=np.random.get_state();assert all(np.array_equal(x,y) for x,y in zip(before,after));assert torch.equal(t,torch.get_rng_state())

def test_model_constructor_preserves_dropout_stream():
    torch.manual_seed(3);state=torch.get_rng_state();model('gcn');assert torch.equal(state,torch.get_rng_state())

def test_validation_depth_selection_and_pdf_diagnosis():
    assert select_depth([{'depth':2,'val':.8,'test':.1},{'depth':24,'val':.7,'test':.9}])==2
    assert classify_depth(.95,1,.2,.3,.1,.8)=='undertrained'
    assert classify_depth(.3,7,.01,.3,.1,.8).startswith('gradient_imbalance')
    assert classify_depth(.3,1,.2,.3,.1,.8)=='overfitting'
    assert classify_depth(.3,1,.01,.3,.1,.8)=='smoothing'
    assert classify_depth(.7,1,.01,.3,.1,.8)=='mixed'

def test_mandatory6_test_label_mutation_real_grid(tmp_path):
    g=load_graph('data/lab5/original.pt');mut={k:v.clone() for k,v in g.items()}
    mut['y'][mut['test_mask']]=torch.randint(0,7,(1000,),generator=torch.Generator().manual_seed(777))
    paths=[]
    for name,graph in [('base',g),('mutant',mut)]:
        p=tmp_path/(name+'.pt');torch.save(graph,p);paths.append(p)
    outcomes=[]
    for p in paths:
        rows=[]
        for lr in [.01,.005]:
            c=config(purpose='test');c['train']['epochs']=3;c['data']['graph_path']=str(p);c['model']['graph_path']=str(p)
            c['optimizer']['lr']=lr;c['logging']['run_root']=str(tmp_path/'runs')
            run=ExperimentRunner(c).run();s=json.loads((run/'summary.json').read_text());weights=torch.load(run/'checkpoints/best.pt',weights_only=False)['model']
            rows.append((s['best_accuracy'],s['best_epoch'],weights))
        outcomes.append(rows)
    for a,b in zip(*outcomes):
        assert a[:2]==b[:2] and all(torch.equal(a[2][k],b[2][k]) for k in a[2])
    assert max(range(2),key=lambda i:outcomes[0][i][0])==max(range(2),key=lambda i:outcomes[1][i][0])

def test_graph_resume_exact_and_unique_runs(tmp_path):
    c=config(purpose='test');c['train']['epochs']=4;c['logging']['run_root']=str(tmp_path/'runs')
    full=ExperimentRunner(c).run();part=ExperimentRunner(c).run(stop_after_epoch=2)
    resumed=ExperimentRunner(c).run(resume=part/'checkpoints/last.pt')
    assert len({full.name,part.name,resumed.name})==3
    a=torch.load(full/'checkpoints/last.pt',weights_only=False);b=torch.load(resumed/'checkpoints/last.pt',weights_only=False)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert a['global_step']==b['global_step']==4
