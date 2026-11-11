"""Small genuine forward/backward demonstrations using the course's unchanged models."""
from pathlib import Path
import argparse,hashlib,json,sys,os,time,uuid,copy
HERE=Path(__file__).resolve().parent;KIT=HERE.parent;SNAP=KIT/'源码快照'
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'新演示记录'/'matplotlib'))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True;sys.path.insert(0,str(SNAP))
import torch
torch.set_num_threads(2)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.small_cnn import build_model

def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def signature(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def fixture(name):return torch.load(HERE/'data'/name,map_location='cpu',weights_only=False)
def run(n):
    start=time.perf_counter();torch.manual_seed(20261111)
    meta=json.loads((HERE/'models.json').read_text('utf-8'))[str(n)];cp=HERE/meta['checkpoint']
    assert sha(cp)==meta['checkpoint_sha256'],'Checkpoint hash mismatch'
    cfg=copy.deepcopy(meta['config']);mc=cfg['model'];mc['graph_path']=None if mc['name']=='graph_network' else mc.get('graph_path')
    # Optional graph_path is ignored by every other registered model.
    model=build_model(mc).eval();state=torch.load(cp,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);before=signature(model);del state
    rid=f'DEMO_lab{n}_{uuid.uuid4().hex}';out=HERE/'新演示记录'/rid;out.mkdir(parents=True)
    result={'RUN_ID':rid,'purpose':'teaching_demo_not_formal_experiment','lab':n,'source_training_RUN_ID':meta['training_run'],'source_evaluation_RUN_ID':meta['evaluation_run'],'source_epoch':meta['source_epoch'],'source_step':meta['source_step'],'checkpoint_sha256':meta['checkpoint_sha256'],'device':'cpu','parameter_updates':0}
    if n in {0,1,2,4,8}:
        d=fixture('cifar16.pt');x,y=d['x'],d['y'];result.update(data='CIFAR-10 first 16 fixed test images',indices=d['indices'])
    if n in {0,2}:
        with torch.no_grad():p=model(x).softmax(-1)
        pred=p.argmax(-1);result.update(predicted=pred.tolist(),true=y.tolist(),demo_correct=int(pred.eq(y).sum()),demo_n=len(y),confidence=p.max(-1).values.tolist())
        names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        fig,axes=plt.subplots(4,4,figsize=(8,8))
        for i,ax in enumerate(axes.flat):
            ax.imshow(d['raw'][i].permute(1,2,0));ax.set_title(f'{i}: {names[pred[i]]}\ntrue: {names[y[i]]}',fontsize=8,color='green' if pred[i]==y[i] else 'darkred');ax.axis('off')
        fig.suptitle(f'Lab {n}: real checkpoint inference; 16-example teaching demo');fig.tight_layout();fig.savefig(out/'predictions.png',dpi=140);plt.close(fig)
    elif n==1:
        logits=model(x);loss=torch.nn.functional.cross_entropy(logits,y);loss.backward()
        gradients={name:float(m.weight.grad.norm()/m.weight.numel()**.5) for name,m in model.named_modules() if isinstance(m,torch.nn.Linear) and m.weight.grad is not None}
        result.update(one_batch_loss=float(loss.detach()),normalized_weight_gradient=gradients,scope='One backward pass, no optimizer step; not the formal cross-step median GI')
        fig,ax=plt.subplots(figsize=(9,4));ax.plot(range(len(gradients)),list(gradients.values()),'o-');ax.set_yscale('log');ax.set(xlabel='Parameterized linear layer',ylabel='RMS weight gradient',title='Lab 1: one real backward pass, no parameter updates');fig.tight_layout();fig.savefig(out/'gradients.png',dpi=140);plt.close(fig);model.zero_grad(set_to_none=True)
    elif n==3:
        d=fixture('copy8.pt');x,y=d['x'],d['y']
        with torch.no_grad():logits=model(x);pred=logits.argmax(1)
        target=y!=-100;result.update(data='Copy test generator, seed 1, first 8 sequences, L=128',correct=int(((pred==y)&target).sum()),target_tokens=int(target.sum()),source_symbols=x[0,:64].tolist(),predicted_copy=pred[0,64:].tolist())
        fig,ax=plt.subplots(figsize=(11,2.4));ax.imshow(torch.stack([x[0,:64],pred[0,64:]]).numpy(),aspect='auto',vmin=0,vmax=63,cmap='tab20');ax.set_yticks([0,1],['Source','Predicted copy']);ax.set_title('Lab 3: copying symbols with a trained sinusoidal model');fig.tight_layout();fig.savefig(out/'copy.png',dpi=140);plt.close(fig)
    elif n==4:
        with torch.no_grad():h,z=model.representations(x);hf,zf=model.representations(x.flip(-1))
        result.update(h_shape=list(h.shape),z_shape=list(z.shape),projection_norm_mean=float(z.norm(dim=1).mean()),paired_flip_cosine=float(torch.nn.functional.cosine_similarity(z,zf).mean()),scope='Frozen SimCLR encoder; this demo is not a trained linear probe or a formal accuracy measurement')
        fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow((z@z.T).numpy(),vmin=-1,vmax=1,cmap='coolwarm');ax.set_title('Lab 4: projected feature similarities');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'similarity.png',dpi=140);plt.close(fig)
    elif n==5:
        d=fixture('cora.pt')
        with torch.no_grad():p=model(d['x'],d['edge_index']).softmax(-1)
        m=d['test_mask'];result.update(nodes=len(d['x']),edges=int(d['edge_index'].shape[1]),test_n=int(m.sum()),test_accuracy=float(p.argmax(-1)[m].eq(d['y'][m]).float().mean()),scope='Entire original Cora graph and fixed public test mask; checkpoint selected historically on val')
        fig,ax=plt.subplots(figsize=(6,3));ax.bar(range(7),torch.bincount(p.argmax(-1)[m],minlength=7).tolist());ax.set(xlabel='Predicted class',ylabel='Test nodes',title='Lab 5: real GCN predictions');fig.tight_layout();fig.savefig(out/'graph_predictions.png',dpi=140);plt.close(fig)
    elif n==6:
        with torch.no_grad():samples=model.sample(256,torch.Generator().manual_seed(20261111));z,_=model(samples);reconstructed=model.inverse(z)
        result.update(generated=256,maximum_inverse_error=float((samples-reconstructed).abs().max()),scope='256 new Flow samples; not the formal 5000-example MMD evaluation')
        fig,ax=plt.subplots(figsize=(5,5));ax.scatter(*samples.T.numpy(),s=8,alpha=.7);ax.set(xlim=(-5,5),ylim=(-5,5),title='Lab 6: 256 newly generated Flow samples');ax.set_aspect('equal');fig.tight_layout();fig.savefig(out/'flow_samples.png',dpi=140);plt.close(fig)
    elif n==7:
        from src.diffusion.schedules import make_schedule
        from src.diffusion.samplers import ddim_sample
        from src.data.gmm2d import inverse_standardize
        noise=torch.randn(256,2,generator=torch.Generator().manual_seed(20261111))
        samples,info=ddim_sample(model,make_schedule(cfg['diffusion']),noise,50,eta=0.)
        raw=inverse_standardize(samples,meta['standardization'])
        result.update(generated=256,NFE=info['NFE'],K=50,T=cfg['diffusion']['T'],eta=0.,sampler_seconds=info['seconds'],scope='256 new DDIM samples with the formal T; not the formal 5000-example quality evaluation')
        fig,ax=plt.subplots(figsize=(5,5));ax.scatter(*raw.T.numpy(),s=8,alpha=.7);ax.set(xlim=(-1.6,1.6),ylim=(-1.6,1.6),title='Lab 7: new DDIM samples, T=1000, K=50');ax.set_aspect('equal');fig.tight_layout();fig.savefig(out/'diffusion_samples.png',dpi=140);plt.close(fig)
    elif n==8:
        with torch.no_grad():logits=model(x);p=logits.softmax(-1);pt=(logits/meta['temperature']).softmax(-1)
        result.update(temperature=meta['temperature'],changed_classes=int(p.argmax(-1).ne(pt.argmax(-1)).sum()),confidence_before=p.max(-1).values.tolist(),confidence_after=pt.max(-1).values.tolist(),scope='Temperature reused from historical val fitting; never refitted on the 16 demo examples')
        fig,ax=plt.subplots(figsize=(8,3));ax.plot(p.max(-1).values,'o-',label='T=1');ax.plot(pt.max(-1).values,'o-',label=f'T={meta["temperature"]:.3f}');ax.set(xlabel='Fixed test image',ylabel='Confidence',title='Lab 8: same predicted classes, changed confidence');ax.legend();fig.tight_layout();fig.savefig(out/'temperature.png',dpi=140);plt.close(fig)
    elif n==9:
        d=fixture('multimodal16.pt');x=d['x'];dv=mc['visual_dim'];da=mc['audio_dim'];v=x[:,:dv];a=x[:,dv:dv+da];prob={}
        with torch.no_grad():
            for name,mask in [('full',(1,1)),('no_audio',(1,0)),('no_video',(0,1))]:prob[name]=model(visual=v,audio=a,mask=mask)
        result.update(data='First 16 fixed RAVDESS test clips; historically normalized pooled encoder features',clip_ids=d['clip_ids'],true=d['y'].tolist(),predictions={k:p.argmax(-1).tolist() for k,p in prob.items()},scope='Live fusion head inference on real frozen features; this entry does not rerun audiovisual preprocessing')
        fig,ax=plt.subplots(figsize=(8,3));im=ax.imshow(torch.stack([p.argmax(-1) for p in prob.values()]).numpy(),aspect='auto',vmin=-.5,vmax=7.5,cmap=plt.get_cmap('tab10',8));ax.set_yticks(range(3),list(prob));ax.set(xlabel='Fixed test clip',title='Lab 9: full pair and missing modality predictions');fig.colorbar(im,ax=ax,ticks=range(8),label='Emotion class');fig.tight_layout();fig.savefig(out/'fusion.png',dpi=140);plt.close(fig)
    result['model_state_unchanged']=signature(model)==before;assert result['model_state_unchanged']
    result['seconds']=time.perf_counter()-start
    result['images']=[p.name for p in out.glob('*.png')];save(out/'result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True);print('\n结果 / Результаты:',out,flush=True)
    return result

def training():
    # Use the original harness with genuine CIFAR train data, isolated demo purpose/output.
    project=Path((HERE/'原项目位置.txt').read_text('utf-8').strip())
    if not (project/'data').is_dir():raise RuntimeError('Short training requires the original local project data. Portable inference entries need no original data.')
    meta=json.loads((HERE/'models.json').read_text('utf-8'))['0'];cfg=copy.deepcopy(meta['config'])
    for k in ['root','split_path']:cfg['data'][k]=str((project/cfg['data'][k]).resolve())
    assert Path(cfg['data']['split_path']).is_file()
    cfg['data'].update(download=False,train_fraction=.003,num_workers=0)
    cfg['experiment'].update(purpose='teaching_demo',role='short_training_demo',name='defense_short_training')
    cfg['train'].update(epochs=1,amp=False);cfg['train'].pop('max_steps',None)
    cfg['runtime'].update(device='cpu',threads=2,max_seconds=90,dependency_lock=str(SNAP/'requirements.lock.txt'))
    cfg['scheduler']['warmup_epochs']=0;cfg['logging']['run_root']=str(HERE/'短训练新记录')
    old=Path.cwd()
    print('教学短训 / Учебный короткий запуск: fresh initialization, one epoch, 0.3% train. Not a formal course result.',flush=True)
    try:
        os.chdir(SNAP)
        from src.engine.runner import ExperimentRunner
        result=ExperimentRunner(cfg).run()
    finally:os.chdir(old)
    print(result,flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--lab',type=int,choices=range(10));p.add_argument('--all',action='store_true');p.add_argument('--train-demo',action='store_true');a=p.parse_args()
    if a.train_demo:training();return
    if a.all:
        results=[run(n) for n in range(10)];save(HERE/'演示核查.json',{'all_passed':True,'models_unchanged':all(r['model_state_unchanged'] for r in results),'seconds':sum(r['seconds'] for r in results),'runs':results});return
    if a.lab is not None:run(a.lab);return
    while True:
        print('\n0–9: 选择实验 / Выберите ЛР; A: 全部演示 / все; T: Lab0真实短训 / короткое обучение; Q: 退出 / выход')
        choice=input('> ').strip().lower()
        if choice=='q':return
        if choice=='a':
            for n in range(10):run(n)
        elif choice=='t':training()
        elif choice.isdigit() and 0<=int(choice)<=9:run(int(choice))
        else:print('请输入0–9、A、T或Q。')

if __name__=='__main__':main()
