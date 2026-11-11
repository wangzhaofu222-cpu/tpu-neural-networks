from pathlib import Path
import json,hashlib,zipfile,sys,subprocess,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'答辩资料_Lab0-9_20261111';KIT=OUT/'05_实验过程';LIVE=KIT/'现场演示'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf-8')

def main():
    import torch
    allcheck=json.loads((LIVE/'演示核查.json').read_text('utf-8'));assert allcheck['all_passed'] and allcheck['models_unchanged']
    modelmeta=json.loads((LIVE/'models.json').read_text('utf-8'))
    for m in modelmeta.values():assert sha(LIVE/m['checkpoint'])==m['checkpoint_sha256']
    history=json.loads((KIT/'历史记录复制校验.json').read_text('utf-8'))
    for row in history:assert sha(ROOT/row['source'])==sha(OUT/row['copy'])==row['sha256'],row['source']
    original_hashes=json.loads((OUT/'04_源码与核查/original_hashes.json').read_text('utf-8'))
    assert all(sha(ROOT/p)==h for p,h in original_hashes.items())
    # Check all existing report/guide PDFs against the previously delivered archive.
    unchanged=0
    with zipfile.ZipFile(ROOT/(OUT.name+'.zip')) as z:
        for folder in ['01_俄语正式报告','02_中文对照译本','03_中俄答辩提纲']:
            for p in (OUT/folder).glob('*.pdf'):
                assert hashlib.sha256(z.read(OUT.name+'/'+p.relative_to(OUT).as_posix())).hexdigest()==sha(p);unchanged+=1
    assert unchanged==21
    train_runs=[p for p in (LIVE/'短训练新记录').iterdir() if p.is_dir() and (p/'summary.json').exists()]
    assert len(train_runs)==1
    p=train_runs[0];summary=json.loads((p/'summary.json').read_text('utf-8'));manifest=json.loads((p/'manifest.json').read_text('utf-8'))
    assert summary['status']=='COMPLETED' and manifest['purpose']=='teaching_demo'
    assert summary['global_step']==2 and summary['epoch']==1 and manifest['actual_train_size']==120
    state=torch.load(p/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert state['run_id']==p.name and state['global_step']==2
    assert all(k in state for k in ['optimizer','scheduler','scaler','python_rng','numpy_rng','torch_rng','loader_rng'])
    cfg=modelmeta['0']['config']
    from importlib import import_module
    sys.path.insert(0,str(KIT/'源码快照'))
    seed_all=import_module('src.utils.reproducibility').seed_all
    build_model=import_module('src.models.small_cnn').build_model
    seed_all(cfg['experiment']['seed'],True,2);initial=build_model(cfg['model']).state_dict()
    max_change=max(float((initial[k]-v).abs().max()) for k,v in state['model'].items() if torch.is_floating_point(v))
    assert max_change>0
    demo_audit={'registered_records':361,'copied_original_text_files':len(history),'text_hashes_match':True,'unchanged_previous_PDFs':21,'checkpoint_files':len(set(m['checkpoint'] for m in modelmeta.values())),'checkpoint_hashes_and_loading':'PASS','live_lab_demonstrations':10,'live_demo_compute_seconds':allcheck['seconds'],'live_models_unchanged':True,'short_training':{'RUN_ID':p.name,'purpose':'teaching_demo','epochs':1,'actual_train_images':120,'parameter_updates':2,'epoch_seconds':summary['epoch_time'],'weight_change_maximum':max_change,'checkpoint_full_state_load':'PASS','path':p.relative_to(OUT).as_posix()},'launcher':'PASS: Windows PowerShell 5 launcher, modes 3 and all, exit code 0','offline_page':json.loads((KIT/'页面功能核查.json').read_text('utf-8')),'plot_visual_review':'PASS: all ten outputs inspected; final Lab4 and Lab9 plots rechecked','formal_results_modified':False,'scope':'Selected checkpoints and real data snippets; complete original dataset, other model checkpoints and Git remain in the original project.'}
    save(KIT/'实验展示最终核查.json',demo_audit)
    short=f'''# 已实际执行的教学短训

RUN_ID：{p.name}

- 调用课程原ExperimentRunner，purpose=teaching_demo。
- CIFAR原train的分层小子集：120图，1轮，2次真实参数更新。
- 完整epoch含val评价：{summary['epoch_time']:.3f}秒。
- 初始与训练后浮点状态最大变化：{max_change:.9f}，证明确实更新了参数。
- 完整checkpoint包含优化器、调度器和随机状态，已重新加载检查。
- 所有输出在{p.relative_to(OUT).as_posix()}。

这是教学短实验，不替代正式12轮Lab0结果。重复运行菜单T会创建新的RUN_ID，不覆盖本次记录。

Это учебный запуск: 120 train-изображений, одна эпоха, два обновления. Он не заменяет формальную ЛР0. Повтор T создаст новый RUN_ID.
'''
    (KIT/'教学短训实跑说明.md').write_text(short,'utf-8')
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='文件清单_SHA256.json']
    manifest_files=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    save(OUT/'文件清单_SHA256.json',manifest_files)
    archive=ROOT/'答辩资料_Lab0-9_含实验展示_20261111.zip';assert not archive.exists()
    print('Packaging',len(files)+1,'files,',round(sum(p.stat().st_size for p in files)/2**20,1),'MiB',flush=True)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*files,OUT/'文件清单_SHA256.json']:z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest_files:
            assert hashlib.sha256(z.read(OUT.name+'/'+row['path'])).hexdigest()==row['sha256']
    result={'folder':str(OUT),'zip':str(archive),'zip_MiB':round(archive.stat().st_size/2**20,2),'CRC_and_all_file_hashes':'PASS',**demo_audit}
    save(ROOT/'tmp/defense_demo_delivery.json',result)
    print(json.dumps({k:result[k] for k in ['zip','zip_MiB','registered_records','checkpoint_files','live_lab_demonstrations','live_demo_compute_seconds','unchanged_previous_PDFs','CRC_and_all_file_hashes']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
