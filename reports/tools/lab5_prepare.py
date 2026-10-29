"""Read-only source ingestion and explicit public-split substitution record."""
import hashlib,json
from pathlib import Path
import torch,yaml
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from src.profile_gen import generate
from src.utils.io import write_json,file_hash
from src.graph.stats import canonical_edges,graph_stats
from src.graph.datasets import fingerprints

def main():
    out=Path('experiments/lab5');out.mkdir(exist_ok=True)
    p=generate('8\u0412\u043c53514');profile=p['profile']
    profile.update({k:p[k] for k in ['lab5_dataset','lab5_second_gnn','lab5_depths']})
    write_json(out/'profile.json',{'profile':profile,'lab5_hash':hashlib.sha256('8\u0412\u043c53514:lab5'.encode()).hexdigest()})
    write_json(out/'split_decision.json',{'date':'2026-10-29','user_decision':'使用 PyG public masks，并保留未核对声明',
        'teacher_protocol_found':False,'teacher_split_independently_verified':False,
        'public_mask_substitution_authorized':True,'cal_role':[]})
    pdf=Path('drive-download-20261026T053112Z-1-001/lab5_graph_representation_learning_1010.pdf')
    write_json(out/'teacher_source.json',{'file':str(pdf),'sha256':file_hash(pdf),'pages':14,'revision':4})
    d=Planetoid('data/planetoid','Cora',split='public',transform=NormalizeFeatures())[0]
    g={k:getattr(d,k).clone() for k in ['x','y','train_mask','val_mask','test_mask']}
    g['edge_index']=canonical_edges(d.edge_index,len(d.y))
    dest=Path('data/lab5');dest.mkdir(exist_ok=True);torch.save(g,dest/'original.pt')
    roles={r:torch.where(g[r+'_mask'])[0].tolist() for r in ['train','val','test']}
    raw=Path('data/planetoid/Cora/raw');testidx=sorted(int(x) for x in (raw/'ind.cora.test.index').read_text().split())
    assert roles['train']==list(range(140)) and roles['val']==list(range(140,640)) and roles['test']==testidx
    assert torch.bincount(g['y'][g['train_mask']]).tolist()==[20]*7
    write_json(out/'data_audit.json',{'status':'PASS_PUBLIC_MASKS_ONLY','teacher_split':'UNVERIFIED',
        'statistics':graph_stats(g['edge_index'],len(g['y']),g['y']),'features':list(g['x'].shape),
        'roles':roles,'immutable_fingerprints':fingerprints(g),'raw_file_sha256':{f.name:file_hash(f) for f in raw.iterdir() if f.is_file()}})
    cfg={'experiment':{'name':'lab5','lab':5,'seed':1,'deterministic':True,'purpose':'development','role':'E1'},
      'data':{'name':'planetoid_graph','graph_path':'data/lab5/original.pt','train_fraction':1.,'num_workers':0},
      'model':{'name':'graph_network','kind':'gcn','input_dim':1433,'classes':7,'width':64,'depth':2,'dropout':.5,
               'residual':False,'graph_path':'data/lab5/original.pt','initialization_seed':1},
      'optimizer':{'name':'adam','lr':.01,'weight_decay':.0005,'momentum':0},
      'scheduler':{'name':'constant','warmup_epochs':0},
      'train':{'epochs':200,'batch_size':2708,'amp':False,'accumulation_steps':1},
      'logging':{'run_root':'runs/lab5','save_every':200,'keep_epochs':[]},
      'runtime':{'device':'cpu','threads':2,'max_seconds':0,'dependency_lock':'docs/environment/lab5-requirements.lock.txt','pause_file':'experiments/lab5/PAUSE'},
      'profile':profile}
    Path('configs/lab5.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    print(json.dumps({'profile':profile,'stats':{k:v for k,v in graph_stats(g['edge_index'],2708,g['y']).items() if k!='degree_sequence'}}))
if __name__=='__main__':main()
