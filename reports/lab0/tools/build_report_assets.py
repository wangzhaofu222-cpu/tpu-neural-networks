"""Derive figures and tables exclusively from the archived Lab0 runs."""
from pathlib import Path
import csv
import json
import math
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/lab0'
SUITE = ROOT / 'experiments/lab0/6116b9614f294c4c918500d92471fc4e'
def read(p): return json.loads(Path(p).read_text('utf-8'))
def lines(p): return [json.loads(x) for x in Path(p).read_text('utf-8').splitlines() if x.strip()]
registry = read(OUT / 'run_registry.json')
runs = {r['condition']: ROOT / 'runs' / r['RUN_ID'] for r in registry}
aliases = {r['condition']: r['alias'] for r in registry}
def epochs(key): return lines(runs[key] / 'metrics.jsonl')
def steps(key): return lines(runs[key] / 'steps.jsonl')
def summary(key): return read(runs[key] / 'summary.json')
def manifest(key): return read(runs[key] / 'manifest.json')
def table(name, header, rows, spec):
    s='\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(header)+' \\\\\n\\midrule\n'
    s+=''.join(' & '.join(map(str,row))+' \\\\\n' for row in rows)
    s+='\\bottomrule\n\\end{tabular}\n'
    (OUT/'tables'/f'{name}.tex').write_text(s,'utf-8')
    with (OUT/'tables'/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
def finish(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.22})
    checks={}
    for r in registry:
        m=manifest(r['condition']); s=summary(r['condition'])
        assert m['RUN_ID']==r['RUN_ID'] and m['EXP_ID']==r['EXP_ID']
        assert m['seed']==r['seed'] and s['validation_metric']==r['accuracy']
    x=epochs('E0'); a,b=x[0],x[-1]
    table('e0',['Показатель','Эпоха 1','Эпоха 12'],[
        ['Train loss',f"{a['train_loss']:.6f}",f"{b['train_loss']:.6f}"],
        ['Val loss',f"{a['validation_loss']:.6f}",f"{b['validation_loss']:.6f}"],
        ['Train accuracy',f"{a['train_metric']:.6f}",f"{b['train_metric']:.6f}"],
        ['Val accuracy',f"{a['validation_metric']:.6f}",f"{b['validation_metric']:.6f}"]], 'lrr')
    fig,axs=plt.subplots(1,2,figsize=(9,3),layout='constrained')
    for mode,col in zip('ABC',['#245a81','#a64b2a','#387657']):
        key=f'E1-{mode}-1'; x=epochs(key); ep=[z['epoch'] for z in x]
        axs[0].plot(ep,[z['train_loss'] for z in x],label=f'{mode}: {aliases[key]}',color=col)
        axs[1].plot(ep,[z['validation_metric'] for z in x],label=mode,color=col)
    axs[0].set(xlabel='Эпоха',ylabel='Средняя train loss');axs[1].set(xlabel='Эпоха',ylabel='Val accuracy')
    for ax in axs: ax.legend();ax.set_xticks([1,3,6,9,12])
    finish(fig,'learning_curves')
    rr=[]; sr=[]; curves={}
    for mode in 'ABC':
        k1,k2=f'E1-{mode}-1',f'E1-{mode}-2'
        m1,m2=manifest(k1),manifest(k2)
        assert m1['EXP_ID']==m2['EXP_ID'] and m1['RUN_ID']!=m2['RUN_ID']
        assert m1['config_hash']==m2['config_hash'] and m1['split_fingerprint']==m2['split_fingerprint']
        assert m1['first_batch_checksum']==m2['first_batch_checksum']
        l1,l2=steps(k1),steps(k2);assert len(l1)==len(l2)
        diff=[abs(a['train_loss']-b['train_loss'])/max(1,abs(a['train_loss'])) for a,b in zip(l1,l2)]
        curves[mode]=diff
        accdiff=abs(summary(k1)['validation_metric']-summary(k2)['validation_metric'])
        rr.append([mode,f'{aliases[k1]}, {aliases[k2]}',str(len(diff)),f'{max(diff):.0f}',f'{accdiff:.0f}'])
        checks[f'E1-{mode}']={'max_relative_loss':max(diff),'accuracy_difference':accdiff}
        kc,ki,kr=f'E3-{mode}-continuous',f'E3-{mode}-interrupted',f'E3-{mode}-resumed'
        lc,lr=steps(kc),steps(kr)
        if len(lr)<len(lc): lr=steps(ki)+lr
        assert [(z['epoch'],z['batch'],z['train_loss'],z['global_step']) for z in lc]==[(z['epoch'],z['batch'],z['train_loss'],z['global_step']) for z in lr]
        cm,rm=manifest(kc),manifest(kr)
        assert cm['global_step']==rm['global_step']
        delta=read(OUT/'report_evidence.json')['checks'][f'E3-{mode}']['delta_theta']
        assert delta==0
        skipped=sum(z.get('optimizer_step_skipped',False) for z in lc)
        sr.append([mode,f'{aliases[kc]}/{aliases[ki]}/{aliases[kr]}',str(cm['global_step']),str(skipped),f'{delta:.0f}',f"{summary(kc)['validation_metric']:.4f}"])
        checks[f'E3-{mode}']={'identical_loss_trajectory':True,'delta_theta_from_archived_check':delta,'optimizer_steps':cm['global_step'],'skipped':skipped}
    table('repeatability',['Режим','Запуски','Mini-batch','Макс. $\\varepsilon_{rel}$','$|\\Delta acc|$'],rr,'llrrr')
    table('resume',['Режим','Непр./прер./resume','Обновления','Пропуски','$\\Delta_\\theta$','Val acc.'],sr,'llrrrr')
    fig,axs=plt.subplots(1,3,figsize=(9,2.6),layout='constrained')
    for ax,mode in zip(axs,'ABC'):
        ax.plot(range(1,len(curves[mode])+1),curves[mode],lw=1.4)
        ax.set(title=f'Режим {mode}',xlabel='Номер mini-batch',ylim=(-.05,.05));ax.set_xticks([1,1008,2016])
    axs[0].set_ylabel('$\\varepsilon_{rel}$ train loss')
    finish(fig,'repeatability')
    values=[summary(f'E2-{i}')['validation_metric'] for i in range(5)]
    mean,sd=st.mean(values),st.stdev(values);half=float(t.ppf(.975,4)*sd/math.sqrt(5))
    archived=read(OUT/'report_evidence.json')['checks']['E2']
    assert abs(mean-archived['mean'])<1e-14 and abs(sd-archived['sample_std'])<1e-14
    assert abs(mean-half-archived['ci95'][0])<1e-12
    checks['E2']={'values':values,'mean':mean,'sample_sd':sd,'ci95':[mean-half,mean+half],'2sd':2*sd}
    table('seeds',['Запуск','Seed','Train loss','Val loss','Val accuracy'],[
        [aliases[f'E2-{i}'],manifest(f'E2-{i}')['seed'],f"{summary(f'E2-{i}')['train_loss']:.6f}",
         f"{summary(f'E2-{i}')['validation_loss']:.6f}",f'{values[i]:.4f}'] for i in range(5)],'lrrrr')
    fig,ax=plt.subplots(figsize=(7.4,2.7),layout='constrained')
    ax.axhspan(mean-half,mean+half,color='#dce7ee',label='95% ДИ среднего')
    ax.axhline(mean,color='#245a81',label='Среднее');ax.scatter(range(5),values,s=45,color='#a64b2a',zorder=3)
    ax.set_xticks(range(5),[aliases[f'E2-{i}'] for i in range(5)])
    ax.set(ylabel='Val accuracy',xlabel='Независимый seed');ax.legend(loc='upper left',fontsize=9)
    finish(fig,'seed_variance')
    fig,axs=plt.subplots(1,3,figsize=(9,2.9),layout='constrained')
    for ax,mode in zip(axs,'ABC'):
        kc,ki,kr=f'E3-{mode}-continuous',f'E3-{mode}-interrupted',f'E3-{mode}-resumed'
        x=epochs(kc);y=epochs(kr)
        if len(y)<len(x): y=epochs(ki)+y
        ax.plot([r['epoch'] for r in x],[r['train_loss'] for r in x],label='Непрерывно',color='#245a81')
        ax.plot([r['epoch'] for r in y],[r['train_loss'] for r in y],'--',label='Полный resume',color='#a64b2a')
        if mode=='A':
            z=epochs('CF03-weights-only');ax.plot([r['epoch'] for r in z],[r['train_loss'] for r in z],':',label='Только веса',color='#387657')
        ax.axvline(6,color='gray',lw=.7);ax.set(title=f'Режим {mode}',xlabel='Эпоха');ax.set_xticks([1,6,12]);ax.legend(fontsize=7.8)
    axs[0].set_ylabel('Train loss');finish(fig,'resume_and_ablation')
    perf={d:read(SUITE/f'E4-{d}-registered.json') for d in ['cpu','cuda']}
    rows=[]
    for dev,d in perf.items():
        for r in d['measurements']:
            samples=r['samples_seconds'];assert len(samples)==100 and r['warmup']==50
            assert abs(st.mean(samples)-r['mean_seconds'])<1e-12
            assert abs(r['batch_size']/st.mean(samples)-r['throughput_samples_per_second'])<1e-8
            assert r['cv_acceptable']
            rows.append([dev.upper(),r['batch_size'],f"{1000*r['mean_seconds']:.4f}",f"{1000*r['median_seconds']:.4f}",f"{1000*r['p95_seconds']:.4f}",f"{r['throughput_samples_per_second']:.2f}",f"{100*r['cv']:.2f}",f"{r['peak_gpu_memory_bytes']/2**20:.2f}" if r['peak_gpu_memory_bytes'] else '---'])
    table('performance',['Устр.','$B$','Сред., мс','Мед., мс','$p_{95}$, мс','Объект/с','CV, \\%','MiB'],rows,'lrrrrrrr')
    fig,axs=plt.subplots(1,3,figsize=(9,2.9),layout='constrained')
    for dev,d in perf.items():
        r=d['measurements'];bs=[z['batch_size'] for z in r]
        axs[0].plot(bs,[z['median_seconds']*1000 for z in r],'o-',label=dev.upper())
        axs[1].plot(bs,[z['throughput_samples_per_second'] for z in r],'o-',label=dev.upper())
        if dev=='cuda':axs[2].plot(bs,[z['peak_gpu_memory_bytes']/2**20 for z in r],'o-',label='CUDA')
    for ax,label in zip(axs,['Медиана batch, мс','Объектов в секунду','Пиковая память, MiB']):
        ax.set(xlabel='Batch size B',ylabel=label,xscale='log');ax.set_xticks([1,16,64,256],['1','16','64','256']);ax.legend(fontsize=9)
    axs[0].set_yscale('log');axs[1].set_yscale('log');finish(fig,'performance')
    erows=[]
    for mode in 'ABC':
        key=f'E1-{mode}-1';z=epochs(key);dur=[r['epoch_time'] for r in z]
        import numpy as np
        peak=max((r.get('peak_gpu_memory_bytes') or 0) for r in z)/2**20
        erows.append([f'{mode} / {aliases[key]}',f'{st.mean(dur):.3f}',f'{np.quantile(dur,.95):.3f}',f'{peak:.2f}' if peak else '---'])
    table('epoch_cost',['Режим / запуск','Средняя эпоха, с','$p_{95}$ эпохи, с','Память, MiB'],erows,'lrrr')
    extra=read(SUITE/'E4-cuda-warmed.json');rows=[]
    for p,e in zip(perf['cuda']['measurements'],extra['measurements']):
        rows.append([e['batch_size'],f"{p['mean_seconds']*1000:.4f}",f"{e['mean_seconds']*1000:.4f}",f"{e['cv']*100:.2f}",'да' if e['cv_acceptable'] else 'нет'])
    table('timing_stability',['$B$','Среднее N=100, мс','Среднее N=300, мс','CV N=300, \\%','CV $\\leq$ 10\\%'],rows,'rrrrc')
    fig,ax=plt.subplots(figsize=(7.4,2.4),layout='constrained')
    for b in [1,16]:
        r=next(r for r in extra['measurements'] if r['batch_size']==b)
        ax.plot(range(1,301),[s*1000 for s in r['samples_seconds']],lw=.75,label=f'B={b}')
    ax.set(xlabel='Номер измерения после прогрева',ylabel='Batch latency, мс');ax.legend(ncols=2)
    finish(fig,'timing_stability')
    reg=['\\begin{longtable}{@{}l>{\\raggedright\\arraybackslash}p{3.7cm}p{10.8cm}@{}}',
         '\\caption{Обозначения запусков и соответствующие RUN\\_ID.}\\label{tab:runs}\\\\',
         '\\toprule Код & Условие & RUN\\_ID \\\\ \\midrule \\endfirsthead',
         '\\toprule Код & Условие & RUN\\_ID \\\\ \\midrule \\endhead',
         '\\bottomrule\\endfoot']
    diagnostic_rows = [
        ('D01', 'Оценка бюджета', 'e592bd6f1f2da6e9f959-f1f208b3b6844b71a89d774e5e291a44'),
        ('D02', 'Начальный E0, K=10', '187f3bd4a93cd598f53b-47b8eee4daa145949b8406914abff222'),
        ('D03', 'Один шаг: состояние AdamW', '699083a5d60b457b7a27-4565dcc8697040c498b47d8fedaae4d4'),
        ('D04', 'Данные до изменения байта', '86c8dd0011a17e664ca2-d4864d8b3e2446ebbd586261f081c1b9'),
        ('D05', 'Данные после изменения байта', '26fbdb25ae58f6ba17cc-acc3b6459ddd475885825852351b1aa8'),
        ('D06', 'E4, CPU, N=100', perf['cpu']['RUN_ID']),
        ('D07', 'E4, CUDA, N=100', perf['cuda']['RUN_ID']),
        ('D08', 'E4, CUDA, N=300', extra['RUN_ID']),
    ]
    all_registry = registry + [{'alias':a, 'condition':c, 'RUN_ID':r} for a,c,r in diagnostic_rows]
    for r in all_registry:
        condition=r['condition'].replace('_',r'\_')
        reg.append('\\hypertarget{'+r['alias']+'}{'+r['alias']+'} & '+condition+
                   ' & {\\fontsize{9.2}{11}\\selectfont\\texttt{'+r['RUN_ID']+'}} \\\\')
    reg.append('\\end{longtable}')
    (OUT/'tables/run_registry.tex').write_text('\n'.join(reg),'utf-8')
    # Record recomputed quantities and their source paths outside the formal prose.
    cf=read(ROOT/'docs/environment/checkpoint-state-probe.json')
    checks['CF03']=cf
    checks['CF04']=read(SUITE/'CF04-extra.json')
    checks['cold_ratio_cpu']=perf['cpu']['measurements'][0]['cold_latency_seconds']/perf['cpu']['measurements'][0]['median_seconds']
    checks['cold_ratio_cuda']=perf['cuda']['measurements'][0]['cold_latency_seconds']/perf['cuda']['measurements'][0]['median_seconds']
    checks['registry_count']=len(registry)
    (OUT/'build/recomputed_report_values.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),'utf-8')
    print('Recomputed 27-run registry, 6 figures and 8 result tables; no training executed.')

if __name__=='__main__':main()
