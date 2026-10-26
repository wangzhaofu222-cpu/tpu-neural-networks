"""Postprocess only saved measurements. This tool never trains or edits run logs."""
import argparse
import json
import math
import re
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'.cache'/'matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(path):
    return [json.loads(x) for x in (Path(path)/'metrics.jsonl').read_text().splitlines()]


def steps(path):
    return [json.loads(x) for x in (Path(path)/'steps.jsonl').read_text().splitlines()]


def n(value,digits=6):
    return f'{value:.{digits}g}' if isinstance(value,(int,float)) else str(value)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('reports/lab0_analysis'))
    args=p.parse_args()
    plt.rcParams.update({'font.size':12,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10})
    ledger=read(args.suite/'ledger.json')
    if not ledger['status'].startswith('TRAINING_PROGRAM_COMPLETED'):
        raise SystemExit('Wait for the complete measured program before writing results')
    args.out.mkdir(parents=True,exist_ok=True)
    figures=args.out/'images'
    figures.mkdir(exist_ok=True)
    checks=ledger['checks']
    all_runs={'E0':Path(ledger['E0']),**{k:Path(v) for k,v in ledger['runs'].items()}}
    extra_path=args.suite/'CF04-extra.json'
    if extra_path.exists():
        extra=read(extra_path)
        for i,path in enumerate(extra['runs']): all_runs[f'CF04-extra-{i+1}']=Path(path)
        if extra['pairs']: checks['CF04']['pairs']=extra['pairs']
    aliases={key:f'R{i:02d}' for i,key in enumerate(all_runs,1)}
    registry=[]
    for key,path in all_runs.items():
        man=read(path/'manifest.json'); summary=read(path/'summary.json')
        registry.append({'alias':aliases[key],'condition':key,'RUN_ID':path.name,'EXP_ID':man['EXP_ID'],
                         'seed':man['seed'],'status':man['status'],'accuracy':summary['validation_metric'],
                         'loss':summary['train_loss'],'epoch':summary['epoch'],'seconds':man['wall_seconds']})
    (args.out/'run_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
    # Learning curves come directly from real epoch logs.
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for key in ['E0','E1-A-1','E1-A-2']:
        rows=metrics(all_runs[key]); epochs=[r['epoch'] for r in rows]
        axes[0].plot(epochs,[r['train_loss'] for r in rows],label=f'{aliases[key]}: {key}')
        axes[1].plot(epochs,[r['validation_metric'] for r in rows],label=f'{aliases[key]}: {key}')
    axes[0].set_ylabel('Training cross entropy'); axes[1].set_ylabel('Validation accuracy')
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend(fontsize=11)
    fig.savefig(figures/'learning_curves.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
    for ax,mode in zip(axes,['A','B','C']):
        a=steps(all_runs[f'E1-{mode}-1']); b=steps(all_runs[f'E1-{mode}-2'])
        loss_a=np.array([r['train_loss'] for r in a]); loss_b=np.array([r['train_loss'] for r in b])
        ax.plot(np.arange(len(a)),np.abs(loss_a-loss_b)/np.maximum(1,np.abs(loss_a)))
        ax.set_title(f'Mode {mode}: {aliases[f"E1-{mode}-1"]}/{aliases[f"E1-{mode}-2"]}')
        ax.set_xlabel('Mini-batch'); ax.set_ylabel('Relative loss difference'); ax.grid(alpha=.2)
    fig.savefig(figures/'repeatability.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(10,3.2),layout='constrained')
    for ax,mode in zip(axes[:3],['A','B','C']):
        for suffix,style in [('continuous','-'),('resumed','--')]:
            key=f'E3-{mode}-{suffix}'; rows=metrics(all_runs[key])
            ax.plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],style,label=aliases[key])
        ax.set_title(f'Resume mode {mode}'); ax.set_xlabel('Epoch'); ax.set_ylabel('Train loss'); ax.set_xticks([1,6,12]); ax.legend(fontsize=10); ax.grid(alpha=.2)
    for key in ['E3-A-continuous','CF03-weights-only']:
        rows=metrics(all_runs[key]); axes[3].plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label=aliases[key])
    axes[3].set_title('Model-only ablation'); axes[3].set_xlabel('Epoch'); axes[3].set_xticks([1,6,12]); axes[3].legend(fontsize=10); axes[3].grid(alpha=.2)
    fig.savefig(figures/'resume_and_ablation.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,1.5),layout='constrained'); ax.axis('off')
    labels=['1. Resolve config','2. Data + split','3. Build model','4. Optimizer + schedule',
            '5. ExperimentRunner','6. Validation / evaluate','7. Save artifacts']
    for i,label in enumerate(labels):
        x=.10+(i%4)*.26; y=.78 if i<4 else .25
        ax.text(x,y,label,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.7',fc='#e7eff8',ec='#345879'),transform=ax.transAxes)
        if i<6:
            nx=.10+((i+1)%4)*.26; ny=.78 if i+1<4 else .25
            if i==3:
                ax.plot([x,x,nx],[.62,.50,.50],color='#345879',lw=1,transform=ax.transAxes)
                ax.annotate('',xy=(nx,.39),xytext=(nx,.50),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#345879'))
            else:
                ax.annotate('',xy=(nx-.10,ny),xytext=(x+.10,y),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#345879'))
    fig.savefig(figures/'harness.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,1.6),layout='constrained'); ax.axis('off')
    ax.text(.02,.80,r'$\bar m=\frac{1}{N}\sum_{i=1}^{N}m_i,\quad s_m=\sqrt{\frac{\sum_i(m_i-\bar m)^2}{N-1}},\quad CI_{95\%}=\bar m\pm t_{0.975,N-1}\frac{s_m}{\sqrt{N}}$',fontsize=14)
    ax.text(.02,.22,r'$\Delta_\theta=\frac{\|\theta_A-\theta_B\|_2}{\|\theta_A\|_2+10^{-12}},\quad \varepsilon_{rel}(x,y)=\frac{|x-y|}{\max(1,|x|)}$',fontsize=14)
    fig.savefig(figures/'metric_definitions.png',dpi=220); plt.close(fig)
    seed_rows=[next(r for r in registry if r['condition']==f'E2-{i}') for i in range(5)]
    fig,ax=plt.subplots(figsize=(8,2.6),layout='constrained')
    ax.scatter(range(5),[r['accuracy'] for r in seed_rows],label='Measured independent runs')
    s=checks['E2']; ax.axhline(s['mean'],color='black',label='Mean')
    ax.axhspan(*s['ci95'],alpha=.15,label='95% Student t CI of mean')
    ax.set_xticks(range(5),[r['alias'] for r in seed_rows]); ax.set_ylabel('Validation accuracy'); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(figures/'seed_variance.png',dpi=180); plt.close(fig)
    performance={}
    for device in ['cpu','cuda']:
        registered=args.suite/f'E4-{device}-registered.json'
        path=registered if registered.exists() else args.suite/f'E4-{device}.json'
        if path.exists(): performance[device]=read(path)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
    for device,data in performance.items():
        rows=data['measurements']; batches=[r['batch_size'] for r in rows]
        axes[0].plot(batches,[r['median_seconds']*1000 for r in rows],marker='o',label=device)
        axes[1].plot(batches,[r['throughput_samples_per_second'] for r in rows],marker='o',label=device)
        if device=='cuda': axes[2].plot(batches,[r['peak_gpu_memory_bytes']/2**20 for r in rows],marker='o',label=device)
    for ax,label in zip(axes,['Median batch latency (ms)','Samples / second','Peak allocated GPU memory (MiB)']):
        ax.set_xscale('log',base=2); ax.set_xlabel('Batch size'); ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend()
    fig.savefig(figures/'performance.png',dpi=180); plt.close(fig)
    first=read(all_runs['E0']/'manifest.json')
    config=yaml.safe_load((all_runs['E0']/'config.resolved.yaml').read_text(encoding='utf-8'))
    runtime=first['environment']; erows=metrics(all_runs['E0'])
    registry_md='\n'.join(f"| {r['alias']} | {r['condition']} | `{r['RUN_ID']}` | {r['seed']} | {r['accuracy']:.4f} | {r['status']} |" for r in registry)
    sections=[]
    sections.append(f'''# Лабораторная работа 0. Инфраструктура воспроизводимых экспериментов

Автор: Ван Чжаофу. Курс «Нейронные сети». Отчёт построен из сохранённых журналов; обозначения Rxx однозначно раскрываются в реестре RUN_ID в конце документа.

Основание требований: lab0_experiment_harness_1010.pdf, редакция 2.1 от 09.10.2026. Структура из шести разделов соответствует §28 исходного задания.

## 1. Профиль и конфигурация

Профиль `{config['profile']['id']}`, хеш идентификатора `{config['profile']['student_id_hash']}`. Исходный идентификатор не публикуется. Использованы CIFAR-10, доля train {config['data']['train_fraction']}, ширина CNN {config['model']['width']}, глубина {config['model']['depth']}, batch size {config['train']['batch_size']}, AdamW, начальная скорость 0.001, weight decay 0.0001, cosine schedule с одной эпохой прогрева. В окончательной программе K=12, CPU threads=2. Число параметров {first['parameters']} ({aliases['E0']}).

Модель: {config['model']['depth']} слоёв Conv2d 3×3 со stride=1, padding=1 и bias, после каждого ReLU; после первых {min(3,config['model']['depth'])} свёрток MaxPool2d 2×2. Все скрытые каналы равны {config['model']['width']}. Затем global average pooling, flatten и Linear {config['model']['width']}→{config['model']['classes']}. BatchNorm, dropout и аугментации отсутствуют.

Из официальных 50000 обучающих изображений выделены train=43000, val=5000, cal=2000; test=10000 сохраняется отдельно. Для ЛР0 используется {first['actual_train_size']} изображений train. Разбиение стратифицировано, seed=1729; доля применяется только к train. Нормализация рассчитана только по train; cal и test не использовались для обучения или подбора. Учебный файл протокола отсутствует, поэтому нельзя утверждать совпадение индексов с файлом преподавателя. Хеш разбиения `{first['split_fingerprint']}` ({aliases['E0']}).

Окружение: Python {runtime['python'].split()[0]}, PyTorch {runtime['pytorch']}, CUDA {runtime['cuda']}, cuDNN {runtime['cudnn']}. GPU: RTX 3060 Laptop 6 GB. Режим A: CPU/deterministic/без AMP/workers=0; B: CUDA/deterministic/без AMP; C: CUDA с AMP. Workers=2 отдельно обозначен как вмешательство режима C, хотя таблица §12.1 включает эти прогоны в общий CPU-бюджет.

Начальная попытка K=10 не прошла условие уменьшения loss вдвое: 1.9642590621→0.9827822588, отношение 0.500332. RUN_ID `187f3bd4a93cd598f53b-47b8eee4daa145949b8406914abff222` сохранён и не объявлен успешным. После проверки журналов выбрана проверяемая коррекция — K=12 с полным переобучением; профиль и исходные данные не изменены. Это изменение длительности, а не продолжение старой cosine-траектории. Измеренная стоимость эпохи около 24 s, оценка 13·12·24≈3744 s без дополнительных CUDA-проверок; фактическое время серии {ledger.get('wall_seconds',0)/60:.2f} min. Бюджетные прогоны не считаются E0.

## 2. Устройство harness

Семь стадий: resolved config → данные и split → модель → optimizer/scheduler → ExperimentRunner → validation/evaluation → артефакты. Общие модули используются всеми условиями. Параметры находятся в YAML, переопределения сохраняются в resolved config.

![Схема семи стадий harness](images/harness.png)

EXP_ID — первые 20 шестнадцатеричных символов SHA256 канонического описания commit/dirty/source hash, resolved config, data/split fingerprint и seed. RUN_ID дополняет EXP_ID независимым UUID. Окружение отдельно записывается в manifest. Изменение результатов не меняет статус исходного кода: область проверки Git явно ограничена src/tests/scripts/configs/pyproject. Между парными прогонами код не коммитировался. Для грязных разработческих прогонов дополнительно сохраняется снимок исходников.

Manifest содержит оба ID, дату, профиль и student hash, Git commit/dirty/source hash, config/data/split hashes, режим, версии, lock hash, устройство, параметры и путь к resolved config. Малые текстовые результаты и split входят в Git; datasets/checkpoints/plots исключены. Без checkpoint теряются точное состояние оптимизатора и возможность немедленного продолжения; воспроизведение потребует повторного обучения.

Checkpoint содержит model, optimizer, scheduler, epoch, фактический global step, Python/NumPy/Torch/CUDA RNG, generator DataLoader, AMP GradScaler. Восстановление поддерживается на границе полной эпохи. Новый процесс создаёт новый RUN_ID и указывает родительский RUN_ID. При AMP пропущенное из-за overflow обновление не считается шагом оптимизатора. Неравные микробатчи при накоплении взвешиваются по числу объектов.

Логи хранят loss, accuracy, lr, epoch/step/time; отдельно сохраняется loss каждого mini-batch. Профилирование использует синхронизацию CUDA до и после интервала, не менее 50 прогревов и 100 измерений на batch size. Все числа ниже рассчитаны из этих файлов.

## 3. Результаты E0–E4

![Определения статистик и относительных расхождений](images/metric_definitions.png)

E0 ({aliases['E0']}): train loss {erows[0]['train_loss']:.6f}→{erows[-1]['train_loss']:.6f}; отношение {erows[-1]['train_loss']/erows[0]['train_loss']:.6f}; validation accuracy {erows[-1]['validation_metric']:.4f}. Независимый evaluate воспроизвёл итоговую accuracy. Условия приёмки: {checks['E0']['status']}.

![Кривые обучения](images/learning_curves.png)
''')
    rows=[]
    for mode in ['A','B','C']:
        x=checks[f'E1-{mode}']; y=checks[f'E3-{mode}']
        rows.append(f"| {mode} | {aliases[f'E1-{mode}-1']}/{aliases[f'E1-{mode}-2']} | {n(x['max_relative_step_loss'])} | {n(x['accuracy_difference'])} | {x['status']} | {aliases[f'E3-{mode}-continuous']}/{aliases[f'E3-{mode}-resumed']} | {n(y['delta_theta'])} | {y['status']} |")
    sections.append('''E1 и E3:

| Режим | E1 runs | max εrel(loss) | Δaccuracy | E1 | E3 runs | Δθ | E3 |
|---|---|---:|---:|---|---|---:|---|
'''+ '\n'.join(rows)+f'''

В A требуются точные траектории и метрика, Δθ≤1e-7; в B — εrel(loss)≤1e-5, Δaccuracy≤.001 и Δθ≤1e-5. В C численные пороги не заданы: таблица сообщает измерение, а не универсальную гарантию. Во всех режимах проверены split, first-batch checksum и число реальных обновлений.

![Разности траекторий](images/repeatability.png)

![Полное восстановление и усечение checkpoint](images/resume_and_ablation.png)

E2 ({', '.join(aliases[f'E2-{i}'] for i in range(5))}): N={s['n']}, средняя validation accuracy={s['mean']:.6f}, выборочное стандартное отклонение={s['sample_std']:.6f}, 95% Student t CI=[{s['ci95'][0]:.6f}; {s['ci95'][1]:.6f}]. Все пять seeds различны; split и остальные условия фиксированы. Ориентир PDF для минимального эффекта — 2s={2*s['sample_std']:.6f}. {'Разброс превышает 1.5 п.п.; требуется дополнительная диагностика.' if s['sample_std']>.015 else 'Разброс не превышает 1.5 п.п.'}

![Разброс по seeds](images/seed_variance.png)

E4: источник обученных весов — {aliases['E1-A-1']}; raw samples находятся в E4-cpu-registered.json/E4-cuda-registered.json серии; исходные замеры E4-cpu.json/E4-cuda.json также сохранены. Forward-only latency отличается от времени training step; длительность эпох берётся из training logs, а не выводится из forward throughput.
''')
    for device,data in performance.items():
        if data.get('RUN_ID'):
            sections.append(f"Отдельный E4 RUN_ID для {device}: `{data['RUN_ID']}`; источник весов `{data.get('source_RUN_ID')}`. "
                f"Измерений на B: {data['measurements'][0]['repeats']}; прогревов не менее {data['measurements'][0]['warmup']}. "
                +('Перед измерениями GPU дополнительно выполнен измеренный прогрев около секунды на каждом B; это повтор после нестабильной исходной серии.' if data.get('settling_is_additional_to_50_warmup_calls') else 'Входы — явно заданные тензоры нормального распределения формы B×3×32×32; это замеры времени, не оценка accuracy.'))
        table=[f'### E4, {device}\n\n| B | mean ms | median ms | p95 ms | samples/s | CV | peak GPU MiB |\n|---:|---:|---:|---:|---:|---:|---:|']
        for r in data['measurements']:
            memory='n/a' if r['peak_gpu_memory_bytes'] is None else f"{r['peak_gpu_memory_bytes']/2**20:.2f}"
            table.append(f"| {r['batch_size']} | {r['mean_seconds']*1000:.4f} | {r['median_seconds']*1000:.4f} | {r['p95_seconds']*1000:.4f} | {r['throughput_samples_per_second']:.2f} | {r['cv']:.4f} | {memory} |")
        sections.append('\n'.join(table))
        bad=[r['batch_size'] for r in data['measurements'] if not r['cv_acceptable']]
        sections.append(f"\nCV>10% при B={bad}. Число повторов увеличено по сравнению с исходными 30 до {data['measurements'][0]['repeats']}; все замеры сохранены. {'Порог вариативности не достигнут, источник полностью не локализован; это ограничение измерения, а не исправленный результат.' if bad else 'Порог CV соблюдён для всех B.'}\n")
    epoch_table=['Время полной эпохи (обучение и validation), фактические журналы E1:',
                 '\n| Режим / run | mean epoch s | p95 epoch s | peak training GPU MiB |',
                 '|---|---:|---:|---:|']
    for mode in ['A','B','C']:
        key=f'E1-{mode}-1'; rows=metrics(all_runs[key]); times=[r['epoch_time'] for r in rows]
        peak=max((r['peak_gpu_memory_bytes'] or 0) for r in rows)
        epoch_table.append(f"| {mode} / {aliases[key]} | {np.mean(times):.3f} | {np.percentile(times,95):.3f} | {peak/2**20:.2f} |")
    sections.append('\n'.join(epoch_table))
    extended=args.suite/'E4-cuda-warmed.json'
    if extended.exists():
        probe=read(extended)
        initial=read(args.suite/'E4-cuda.json')['measurements']
        registered=read(args.suite/'E4-cuda-registered.json')['measurements']
        table=[f"Дополнительная проверка устойчивости GPU timing: RUN_ID `{probe['RUN_ID']}`, "
               "измеренный прогрев не менее секунды на каждом B, затем 50 дополнительных warmup и 300 измерений. "
               "Она не заменяет предписанную серию выше; все три серии сохранены.",
               '\n| B | initial mean ms / N100 | registered mean ms / N100 | extra mean ms / N300 | extra CV | SM MHz before→after |',
               '|---:|---:|---:|---:|---:|---|']
        for x,y,z in zip(initial,registered,probe['measurements']):
            clocks=[z[k].split(',')[0] if z[k] else 'n/a' for k in ['clocks_before','clocks_after']]
            table.append(f"| {z['batch_size']} | {x['mean_seconds']*1000:.4f} | {y['mean_seconds']*1000:.4f} | {z['mean_seconds']*1000:.4f} | {z['cv']:.4f} | {clocks[0]}→{clocks[1]} |")
        bad=[r['batch_size'] for r in probe['measurements'] if not r['cv_acceptable']]
        table.append(f"\nВ дополнительной серии порог CV не достигнут при B={bad}. Увеличение N до 300 и дополнительный прогрев "
            "не устранили вариативность. Для B16 наблюдалось изменение частоты SM, но по этим данным нельзя приписать все выбросы "
            "частоте или полностью разделить GPU clock variation и задержки запуска на host. Поэтому устойчивость latency вне "
            "конкретной серии не установлена; цифры для малых B имеют это ограничение. Сырые выбросы не исключались.")
        sections.append('\n'.join(table))
    sections.append('![Ресурсные характеристики](images/performance.png)\n\n## 4. Уровни воспроизводимости\n')
    for mode in ['A','B','C']:
        passed=checks[f'E1-{mode}']['status']=='PASS' and checks[f'E3-{mode}']['status']=='PASS'
        if mode=='C':
            passed=passed and checks['E1-C']['max_relative_step_loss']==0 and checks['E1-C']['accuracy_difference']==0 and checks['E3-C']['delta_theta']==0
        level='R3 (наблюдаемое совпадение серии)' if passed else 'R0/R1; R2/R3 не установлены, см. численные расхождения'
        sections.append(f"Для режима {mode} измеренная серия подтверждает {level} в зафиксированном окружении. "+('В C совпадение, если наблюдается, не означает обещания побитовой повторяемости для любых AMP/worker условий.' if mode=='C' else '')+'\n')
    w=checks['CF02']; broken=checks['CF03']; f=checks['CF06']; cpu=performance['cpu']['measurements']
    smallest=min(cpu,key=lambda r:r['batch_size']); latency=min(cpu,key=lambda r:r['median_seconds']); throughput=max(cpu,key=lambda r:r['throughput_samples_per_second'])
    sections.append(f'''## 5. Контрфактические вопросы

1. По E2 среднее={s['mean']:.6f}, s={s['sample_std']:.6f}, ориентир 2s={2*s['sample_std']:.6f} ({', '.join(aliases[f'E2-{i}'] for i in range(5))}). Это эвристика PDF, не рассчитанная мощность теста. При допущении независимых одинаково распределённых прогонов стандартная ошибка среднего убывает как N^(-1/2), поэтому для вдвое меньшего эффекта при фиксированных квантилях требуется примерно 4N=20 прогонов. При ~24s/epoch и K=12 это около 96 min только на одну серию, без CUDA и диагностики. Точное число для заданной мощности требует заранее выбранного теста/альтернативы; из пяти измерений универсальную гарантию получить нельзя.

2. Два запуска workers=2: {aliases['CF02-workers-1']}/{aliases['CF02-workers-2']}; max εrel(loss)={n(w['max_relative_step_loss'])}, Δaccuracy={n(w['accuracy_difference'])}. Сравнение с workers=0 ({aliases['E1-A-1']}): {json.dumps(w['against_workers0'],ensure_ascii=False)}. В текущем CIFAR pipeline нет случайных аугментаций, порядок задаётся отдельным generator, workers инициализированы; поэтому несколько workers сами по себе не обязаны менять результат. Наблюдение относится только к этому pipeline.

3. Только model.state_dict загружен из настоящего усечённого checkpoint, остальные поля физически отсутствуют. Новый результат {aliases['CF03-weights-only']} против {aliases['E3-A-continuous']}: Δθ={broken['delta_theta']:.8g}. Это совокупный эффект потери optimizer moments, scheduler, RNG/порядка batch и состояния продолжения. По одному вмешательству нельзя приписать всю величину одному полю; отдельное разложение по компонентам не выполнялось. Полное восстановление в E3 восстанавливает их совместно.

4. Пары E2 с |Δaccuracy|<.001: {json.dumps(checks['CF04']['pairs'],ensure_ascii=False)}. {('В найденных парах равенство/неравенство траекторий проверено по всем step losses. Близкая конечная accuracy не доказывает совпадения траектории.' if checks['CF04']['pairs'] else 'В пяти предписанных запусках подходящей пары нет. Требуемый частный пример пока не получен; нельзя выдумывать пару или менять её порог. Пункт требует дополнительных seeds и не объявляется завершённым.')}

5. В manifest {aliases['E1-A-1']} сохранён cuDNN={checks['CF05']['cudnn_recorded']}. При замене версии изменится environment.cudnn, но не каноническая функция EXP_ID, если код/config/data/seed прежние. Фактическая замена библиотеки не выполнялась. Происхождение R0 остаётся описываемым, R1 для сохранённого split можно проверить; R2/R3 после изменения окружения без повторных измерений не гарантируются.

6. Один байт изменён только в копии официального data_batch_1. Исходный SHA256={f['original_hash']}; SHA256 копии={f['changed_copy_hash']}; изменение обнаружено={f['detected']} ({aliases['E1-A-1']} как ссылка на исходный data fingerprint). Имя/путь исходника и код/config/seed не менялись; подстановка испорченного файла изменила бы content fingerprint и EXP_ID, а встроенная MD5-проверка CIFAR дополнительно отклонила бы файл до обучения. Исходные данные не повреждались.

7. Для B={smallest['batch_size']} CPU first forward={smallest['cold_latency_seconds']*1000:.4f} ms, median после прогрева={smallest['median_seconds']*1000:.4f} ms, отношение={smallest['cold_latency_seconds']/smallest['median_seconds']:.3f} ({aliases['E1-A-1']} / E4-cpu.json). Первый вызов включает ленивую подготовку вычислений и памяти; отдельная трассировка этих затрат не выполнялась.

8. Минимальная CPU median latency при B={latency['batch_size']}: {latency['median_seconds']*1000:.4f} ms. Максимальный throughput при B={throughput['batch_size']}: {throughput['throughput_samples_per_second']:.2f} samples/s ({aliases['E1-A-1']} / E4-cpu.json). Batch latency и стоимость одного изображения различны; больший batch распределяет накладные расходы по большему числу объектов, пока не насыщается вычислитель.

## 6. Ограничения

Нельзя переносить результат на другую версию PyTorch/CUDA/cuDNN, устройство или недетерминированный pipeline без повторной проверки. Восстановление внутри произвольного mini-batch не реализовано. CPU temperature не измерялась; пользователь просил не задерживать работу из-за отсутствия датчика. Официальная страница CIFAR не указывает отдельную лицензию; исходные изображения не распространяются в репозитории. Учебный протокол и стандартный файл split отсутствуют. Для C совпадение не является универсальной гарантией. Вариативность GPU timing при CV>10% остаётся ограничением; данные не сглаживались и не заменялись. Если CF4 не содержит пары, этот пункт и полная приёмка ЛР0 остаются незавершёнными.

### Реестр запусков

| Обозначение | Условие | RUN_ID | seed | val accuracy | Статус |
|---|---|---|---:|---:|---|
{registry_md}
''')
    text='\n\n'.join(sections)
    worker_words=(f"2. Два запуска workers=2: {aliases['CF02-workers-1']}/{aliases['CF02-workers-2']}; "
        f"max εrel(loss)={n(w['max_relative_step_loss'])}, Δaccuracy={n(w['accuracy_difference'])}. "
        f"При сопоставлении с workers=0 ({aliases['E1-A-1']}) split, checksum первого batch и все "
        f"{len(steps(all_runs['E1-A-1']))} step losses совпали; Δaccuracy={n(w['against_workers0']['final_accuracy_difference'])}. "
        "В текущем pipeline отсутствуют случайные аугментации, порядок задаётся отдельным generator, workers получают фиксированные seed. "
        "Число workers изменяет выполнение загрузки, но в измеренной серии не изменило входные данные и арифметику модели; "
        "это объясняет отсутствие расхождения. Наблюдение не распространяется на произвольные аугментации и другие pipelines.")
    if all(w['against_workers0'][k] for k in ['split_equal','first_batch_equal','step_losses_equal']):
        text=re.sub(r'2\. Два запуска workers=2:.*?(?=\n\n3\.)',lambda _:worker_words,text,flags=re.S)
    if checks['CF04']['pairs']:
        descriptions=[]
        for pair in checks['CF04']['pairs']:
            rr=[next(row for row in registry if row['RUN_ID']==rid) for rid in pair['RUN_IDs']]
            descriptions.append(f"{rr[0]['alias']}/{rr[1]['alias']}: accuracy={rr[0]['accuracy']:.4f}/{rr[1]['accuracy']:.4f}, "
                f"|Δaccuracy|={pair['accuracy_difference']:.6f}, совпадение всех step losses — "+('да' if pair['trajectories_equal'] else 'нет')+
                (f", max εrel(loss)={pair['max_relative_step_loss_difference']:.6f}, различных step losses={pair['unequal_step_losses']}" if 'max_relative_step_loss_difference' in pair else ''))
        explanation=('4. В исходной E2 из пяти seeds пары с |Δaccuracy|<.001 не оказалось. '
            +f"Число дополнительных запусков по тому же протоколу: {len(extra.get('runs',[])) if extra_path.exists() else 0}; дополнительный бюджет был заранее ограничен. "
            +'Проверенные пары: '+'; '.join(descriptions)+'. Основные mean/s/CI выше по-прежнему рассчитаны только по исходным пяти seeds. '
            +'Близкая конечная accuracy не доказывает воспроизведение траектории; дополнительный поиск пары не является оценкой эффекта нового метода.')
        text=re.sub(r'4\. Пары E2.*?(?=\n\n5\.)',lambda _:explanation,text,flags=re.S)
        text=text.replace('Если CF4 не содержит пары, этот пункт и полная приёмка ЛР0 остаются незавершёнными.',
            'Дополнительные seeds CF4 служат только контрпримером: адаптивно найденная пара не заменяет исходную серию E2.')
    text=text.replace('### Реестр запусков','### Реестр обучающих запусков')
    text=text.replace('Первый вызов включает ленивую подготовку вычислений и памяти; отдельная трассировка этих затрат не выполнялась.',
        'Увеличение первого интервала согласуется с ленивой подготовкой вычислений и памяти; без отдельной трассировки разделить эти составляющие нельзя.')
    text=text.replace('пока не насыщается вычислитель.',
        'пока не насыщается вычислитель. Это эмпирический максимум данной серии; превосходство соседних B отдельным статистическим тестом не установлено.')
    text=text.replace('фактическое время серии ', 'фактическое время основного serial-suite (без ранее выполненного E0 и последующих дополнительных CF4/profiling) ')
    text=text.replace('Бюджетные прогоны не считаются E0.',
        'Бюджетный RUN_ID `'+ledger['budget']['RUN_ID']+'`. Бюджетные прогоны не считаются E0. Тестируемый commit основной серии `'+ledger['code_commit']+'`.')
    text=text.replace('В C совпадение, если наблюдается, не означает обещания побитовой повторяемости для любых AMP/worker условий.',
        'В C наблюдаемое совпадение объясняется фиксированными seed, потоком данных, алгоритмами и одним устройством; AMP сам по себе не вводит случайность. Это не обещание побитовой повторяемости для любых AMP/worker условий.')
    if extra_path.exists():
        text=text.replace('Пары E2 с |Δaccuracy|<.001:',
            'Пары с |Δaccuracy|<.001 в исходной E2 и отдельно отмеченных дополнительных seeds CF4 (основной CI по пяти прогонам не пересчитан):')
    text=text.replace('при фиксированных квантилях требуется примерно 4N=20 прогонов.',
        'при фиксированных квантилях требуется примерно 4N=20 прогонов. Дополнительные 15 запусков означают около 72 min сверх принятой оценки всей серии около 62 min; без пересмотра исходного бюджета это не укладывается.')
    component_path=ROOT/'docs/environment/checkpoint-state-probe.json'
    if component_path.exists():
        cp=read(component_path)
        explanation=(f"3. Усечённый checkpoint содержит только model.state_dict; итог {aliases['CF03-weights-only']} против {aliases['E3-A-continuous']}: "
            f"Δθ={broken['delta_theta']:.8g}. Для локализации выполнен отдельный опыт RUN_ID `{cp['RUN_ID']}`: "
            f"одна и та же сохранённая точка и один и тот же градиент (loss={cp['fixed_gradient_loss']:.8g}). "
            f"Сброс только моментов AdamW при прежней lr дал Δθ={cp['delta_theta_reset_optimizer_moments_only']:.8g}; "
            f"изменение только lr с {cp['resumed_lr']:.8g} на {cp['initial_lr']:.8g} при сохранённых моментах дало "
            f"Δθ={cp['delta_theta_reset_learning_rate_only']:.8g}. Без generator state порядок следующего batch меняется: "
            f"{'да' if cp['data_order_changes_without_loader_rng'] else 'нет'}. Loss контрольного batch совпал с реально записанным первым шагом эпохи 7. "
            "Таким образом, потеря моментов, положения schedule и порядка данных даёт независимо проверяемые источники расхождения. "
            "Локальные одношаговые разности нельзя складывать как разложение конечного Δθ. Для данного CNN нет dropout и случайных аугментаций; "
            "Python/NumPy/global Torch RNG сохранены для общего интерфейса, их отдельное влияние здесь не измерялось.")
        text=re.sub(r'3\. Только model\.state_dict.*?(?=\n\n4\.)',lambda _:explanation,text,flags=re.S)
    byte_path=ROOT/'docs/environment/data-byte-probe.json'
    if byte_path.exists():
        bp=read(byte_path)
        fields=bp['field_equality']
        assert fields['config_hash'] and fields['split_fingerprint'] and fields['seed'] and fields['git_commit']
        assert not fields['data_fingerprint'] and not fields['EXP_ID'] and bp['official_data_untouched']
        explanation=(f"6. Выполнены два состояния отдельной sandbox-копии data_batch_1 по одному и тому же имени и пути, размер не изменился. "
            f"Диагностические RUN_ID `{bp['RUN_IDs'][0]}` и `{bp['RUN_IDs'][1]}`. SHA256 до изменения={bp['before_hash']}, "
            f"после изменения одного байта={bp['after_hash']}. Изменились data fingerprint и EXP_ID; config hash, split fingerprint, seed, "
            "Git commit, source hash и профиль остались прежними. У независимых диагностических запусков различаются RUN_ID. "
            "Официальные обучающие файлы не изменялись. MD5-проверка CIFAR дополнительно отклонила бы испорченный файл до обучения.")
        text=re.sub(r'6\. Один байт изменён.*?(?=\n\n7\.)',lambda _:explanation,text,flags=re.S)
    step_details=[]
    for mode in ['A','B','C']:
        cr=read(all_runs[f'E3-{mode}-continuous']/'summary.json')
        rr=read(all_runs[f'E3-{mode}-resumed']/'summary.json')
        skipped=sum(bool(r.get('optimizer_step_skipped')) for r in steps(all_runs[f'E3-{mode}-continuous']))
        step_details.append(f"{mode}: continuous/resume={cr['global_step']}/{rr['global_step']}, пропущено из-за overflow={skipped}")
    text=text.replace('Во всех режимах проверены split, first-batch checksum и число реальных обновлений.',
        'Во всех режимах проверены split, first-batch checksum и число реальных обновлений. '+ '; '.join(step_details)+'.')
    text=text.replace('/ E4-cpu.json','/ E4-cpu-registered.json')
    text=text.replace('### Реестр обучающих запусков',
        'PDF экспортирован из общего текста через ReportLab 4.4.9 с встроенными кириллическими шрифтами. Полный LaTeX-источник сохранён; его компиляция не проверена: процесс встроенного компилятора не запускается, ошибка 1060.\n\n'
        'После основной серии неизменные численности val/cal вынесены в YAML, добавлен табличный вывод compare. Исторические manifest не переписаны. Проверка выпуска отдельно сравнивает данные и первый реальный epoch; полная серия относится к указанному выше исходному commit.\n\n### Реестр обучающих запусков')
    compatibility=ROOT/'docs/environment/release-compatibility.json'
    if compatibility.exists():
        cp=read(compatibility)
        paragraph=(f"Проверка выпуска: RUN_ID `{cp['release_RUN_ID']}`, commit `{cp['release_commit']}`, статус {cp['status']}. "
            f"Data/split/subset fingerprints, нормализация и checksum совпали; loss первого реального epoch={cp['train_loss']:.10f}, "
            f"val accuracy={cp['validation_accuracy']:.4f}, обе величины точно совпали с первым epoch исходной серии. "
            "Это проверка данных и одного epoch, не повтор всей программы E0–E4.\n\n")
        text=text.replace('### Реестр обучающих запусков',paragraph+'### Реестр обучающих запусков')
    import xml.etree.ElementTree as ET
    test_path=ROOT/'docs/environment/tests-lab0-release.xml'
    if test_path.exists():
        test=ET.parse(test_path).getroot().find('testsuite').attrib
        text=text.replace('## 3. Результаты E0–E4',
            f"Проверка выпуска: {test['tests']} автоматических тестов; failures={test['failures']}, errors={test['errors']}, "
            f"skipped={test['skipped']}, {float(test['time']):.3f} s (tests-lab0-release.xml). "
            "Тесты обучения используют маркированные synthetic fixtures и не заменяют реальные эксперименты ниже.\n\n## 3. Результаты E0–E4")
    (args.out/'report.ru.md').write_text(text,encoding='utf-8')
    write_json=args.out/'report_evidence.json'
    write_json.write_text(json.dumps({'suite':str(args.suite),'checks':checks,'run_registry':registry,
                                    'full_course_complete':False,'CF04_complete':bool(checks['CF04']['pairs'])},ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.out/'report.ru.md')


if __name__=='__main__':
    main()
