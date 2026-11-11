"""Assemble authentic evidence and bounded live demonstrations; never alter old runs."""
from pathlib import Path
import re,json,shutil,hashlib,sys,os
import yaml,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'答辩资料_Lab0-9_20261111'
KIT=OUT/'05_实验过程';LIVE=KIT/'现场演示';SNAP=KIT/'源码快照'
TITLES=['可复现实验框架 / Воспроизводимость','梯度与残差 / Градиенты и residual','CNN 与 ViT','位置与长度外推 / Позиция и длина','自监督学习 / SimCLR','图神经网络 / GNN','能量模型与流 / EBM и Flow','扩散模型 / Диффузия','校准与可靠性 / Калибровка','多模态融合 / Fusion']
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,q):
    q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def records(p):
    if not p.exists():return []
    return [json.loads(l) for l in p.read_text('utf-8-sig').splitlines() if l.strip()]
def prepare():
    for p in [KIT,LIVE,SNAP,LIVE/'data',LIVE/'models']:p.mkdir(parents=True,exist_ok=True)
    for directory in ['src','scripts','configs','tests','splits']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.yaml','.yml','.json'}:copy(p,SNAP/p.relative_to(ROOT))
    for directory in ['reports/tools','reports/formal_tools']:
        for p in (ROOT/directory).glob('*.py'):copy(p,SNAP/p.relative_to(ROOT))
    for p in [ROOT/'pyproject.toml',ROOT/'requirements.lock.txt',ROOT/'requirements-lab9.lock.txt']:
        copy(p,SNAP/p.name)
    for p in (ROOT/'docs/environment').glob('*lock*'):
        if p.is_file():copy(p,SNAP/p.relative_to(ROOT))
    by={p.name:p for p in (ROOT/'runs').rglob('*') if p.is_dir() and re.fullmatch('[0-9a-f]{20}-[0-9a-f]{32}',p.name)}
    selected=json.loads((ROOT/'tmp/defense_demo_inventory.json').read_text('utf-8'));models={};labs=[];hashes=[]
    for n in range(10):
        src=ROOT/f'reports/lab{n}'/('formal' if n else '')
        text='\n'.join(p.read_text('utf-8-sig') for p in [src/'report_body.tex',*sorted((src/'tables').glob('*.tex'))])
        registry=[]
        for line in text.splitlines():
            m=re.search(r'\\hypertarget\{([^}]+)\}.*?([0-9a-f]{20}-[0-9a-f]{32})',line)
            if not m:continue
            alias,rid=m.groups();p=by[rid]
            for f in p.iterdir():
                if f.is_file() and f.suffix in {'.yaml','.json','.jsonl','.csv','.txt'}:
                    q=KIT/'真实运行记录'/f'lab{n}'/rid/f.name;copy(f,q)
                    hashes.append({'source':str(f.relative_to(ROOT)),'copy':str(q.relative_to(OUT)),'sha256':sha(f)})
            registry.append({'alias':alias,'run_id':rid,'folder':str((KIT/'真实运行记录'/f'lab{n}'/rid).relative_to(OUT)).replace('\\','/')})
        # Copy compact experimental summaries, protocols, hypotheses and result tables.
        for f in (ROOT/f'experiments/lab{n}').glob('*'):
            if f.is_file() and f.suffix in {'.json','.md','.csv','.yaml'}:copy(f,KIT/'实验汇总与协议'/f'lab{n}'/f.name)
        ev=ROOT/selected[str(n)]['path'];summary=json.loads((ev/'summary.json').read_text('utf-8'));cfg=yaml.safe_load((ev/'config.resolved.yaml').read_text('utf-8'))
        train_id=(summary.get('parent_RUN_ID') if n==3 else summary.get('encoder_RUN_ID') if n==4 else Path(cfg['evaluation_spec']['path']).name if n==9 else summary.get('source_RUN_ID')) or ev.name
        tr=by[train_id];tcfg=yaml.safe_load((tr/'config.resolved.yaml').read_text('utf-8'))
        expected=summary.get('encoder_checkpoint_sha256') if n==4 else summary.get('source_checkpoint_sha256',summary.get('checkpoint_sha256'))
        candidates=list((tr/'checkpoints').glob('*.pt'))
        desired='best.pt' if n in {5,9} else summary.get('checkpoint','last.pt') if n==4 else 'last.pt'
        candidates.sort(key=lambda p:(p.name!=desired,p.name))
        cp=next((p for p in candidates if not expected or sha(p)==expected),None)
        assert cp is not None,(n,train_id,expected)
        q=LIVE/'models'/train_id/cp.name
        if not q.exists():copy(cp,q)
        state=torch.load(cp,map_location='cpu',weights_only=False)
        assert state['run_id']==train_id,(n,'run identity')
        record={'lab':n,'report_alias':selected[str(n)]['alias'],'evaluation_run':ev.name,'training_run':train_id,'checkpoint':str(q.relative_to(LIVE)).replace('\\','/'),'checkpoint_sha256':sha(cp),'config':tcfg,'source_epoch':state['epoch'],'source_step':state['global_step']}
        if n==7:record['standardization']=json.loads((ROOT/'experiments/lab7/data_protocol.json').read_text('utf-8'))['normalization']
        if n==8:record['temperature']=next(r['temperature'] for r in json.loads((ROOT/'experiments/lab8/E1.json').read_text('utf-8'))['rows'] if r['model']=='cnn' and r['seed']==1 and r['temperature']!=1)
        models[str(n)]=record
        # Training logs and manifests are distinct from evaluation records.
        for f in tr.iterdir():
            if f.is_file() and f.suffix in {'.yaml','.json','.jsonl','.txt'}:copy(f,KIT/'训练来源'/f'lab{n}'/train_id/f.name)
        loss=records(tr/'steps.jsonl') or records(tr/'metrics.jsonl')
        points=[{'step':r.get('global_step',r.get('epoch',i+1)),'loss':r.get('train_loss')} for i,r in enumerate(loss) if isinstance(r.get('train_loss'),(float,int))]
        epochs=records(tr/'metrics.jsonl')
        plots=[str(p.relative_to(OUT)).replace('\\','/') for p in sorted((OUT/'04_源码与核查'/f'lab{n}_zh/figures').glob('*.png'))]
        labs.append({'n':n,'title':TITLES[n],'model':record,'runs':registry,'loss':points,'epochs':epochs,'plots':plots})
        del state
    # Real, fixed data examples. No cal data is accessed.
    from torchvision.datasets import CIFAR10
    c0=models['0']['config'];norm=json.loads((by[models['0']['training_run']]/'manifest.json').read_text('utf-8'))['normalization']
    test=CIFAR10(str(ROOT/c0['data']['root']),train=False,download=False)
    x=torch.from_numpy(test.data[:16].copy()).permute(0,3,1,2).float()/255
    torch.save({'raw':x,'x':(x-torch.tensor(norm['mean'])[None,:,None,None])/torch.tensor(norm['std'])[None,:,None,None],'y':torch.tensor(test.targets[:16]),'normalization':norm,'indices':list(range(16)),'role':'test','dataset':'CIFAR-10 official'},LIVE/'data/cifar16.pt')
    copy(ROOT/'data/lab5/original.pt',LIVE/'data/cora.pt')
    from src.data.synthetic_sequence import Sequences
    seq=Sequences('copy',128,1,'test',8,32,128);xy=[seq[i] for i in range(8)]
    torch.save({'x':torch.stack([x for x,y in xy]),'y':torch.stack([y for x,y in xy]),'role':'test','indices':list(range(8)),'seed':1},LIVE/'data/copy8.pt')
    feature=torch.load(ROOT/'data/lab9/features/pooled.pt',weights_only=False,map_location='cpu')['test']
    torch.save({k:v[:16] for k,v in feature.items()},LIVE/'data/multimodal16.pt')
    manifest=json.loads((ROOT/'experiments/lab9/data_manifest.json').read_text('utf-8'))
    clip=next(c for c in manifest['clips'] if c['role']=='test' and c['clip_id']==feature['clip_ids'][0]);copy(ROOT/clip['media_path'],LIVE/'data/ravdess_example.mp4')
    save(LIVE/'data/ravdess_example.json',{k:clip[k] for k in ['clip_id','actor_id','emotion','role','source_sha256','official_archive','archive_entry']})
    save(LIVE/'models.json',models);save(KIT/'展示索引.json',labs);save(KIT/'历史记录复制校验.json',hashes)
    (LIVE/'原项目位置.txt').write_text(str(ROOT),'utf-8')
    # Human explanation is supplied separately; do not copy stale top-level progress snapshots.
    print('Copied',sum(len(x['runs']) for x in labs),'registered records,',len(set(m['checkpoint'] for m in models.values())),'checkpoints; training curves',[(x['n'],len(x['loss'])) for x in labs],flush=True)

if __name__=='__main__':prepare()
