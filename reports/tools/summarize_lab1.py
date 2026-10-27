"""Generate the Lab1 report directly from immutable run records (no fitted data)."""
import hashlib
import itertools
import json
import math
import statistics
import subprocess
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path('tmp/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':12,'xtick.labelsize':11,'ytick.labelsize':11})

ROOT = Path('reports/lab1')
DEPTHS = [8,24,48,96]
ARCHS = ['plain','residual_unit','residual_scaled']
NAMES = {'plain':'MLP', 'residual_unit':r'Res $\alpha=1$', 'residual_scaled':r'Res $\alpha=L^{-1/2}$'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def number(x):
    if x is None:
        return '--'
    if isinstance(x,bool):
        return 'да' if x else 'нет'
    if isinstance(x,int):
        return str(x)
    if x == 0:
        return '0'
    return f'{x:.4g}'


def tex(text):
    return str(text).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')


def table(headers, rows, columns=None):
    columns = columns or 'l'+'r'*(len(headers)-1)
    return '\n'.join([r'\begin{center}\small',r'\begin{tabular}{'+columns+'}',r'\toprule',
                      ' & '.join(headers)+r' \\',r'\midrule',
                      *[' & '.join(map(str,row))+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}\end{center}'])


def figure(name, caption, height='.72'):
    return r'\begin{center}\includegraphics[width=\linewidth,height='+height+r'\textheight,keepaspectratio]{images/'+name+r'.png}\par\small '+caption+r'\end{center}'


def main():
    ROOT.mkdir(exist_ok=True)
    (ROOT/'images').mkdir(exist_ok=True)
    ledger = read('experiments/lab1/ledger.json')
    runs = {}
    for i,r in enumerate(ledger,1):
        path = Path(r['path'])
        runs[r['label']] = {**r,'ref':f'R{i:02}', 'manifest':read(path/'manifest.json'),
            'config':__import__('yaml').safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8')),
            'summary':read(path/'summary.json'), 'initial':read(path/'probe.initial.json')}
        if (path/'probe.final.json').exists():
            runs[r['label']]['final'] = read(path/'probe.final.json')
    from src.compare import compare_runs,markdown_table
    comparison=compare_runs([r['path'] for r in ledger])
    write(ROOT/'comparison.json',comparison)
    (ROOT/'comparison.md').write_text(markdown_table(comparison),encoding='utf-8')
    def ref(label):
        return runs[label]['ref']
    def met(label,key):
        return runs[label]['summary']['trainability'].get(key)
    lr = read('experiments/lab1/selected_lr.json')['lr']
    prereg = read('experiments/lab1/e3_preregistration.json')
    a = [met(f'E3_A_seed{s}','rho_T') for s in [1,2,3]]
    b = [met(f'E3_B_seed{s}','rho_T') for s in [1,2,3]]
    deltas = [x-y for x,y in zip(a,b)]
    mean, sd = statistics.mean(deltas), statistics.stdev(deltas)
    established = (all(d>0 for d in deltas) or all(d<0 for d in deltas)) and abs(mean)>=2*sd
    e1_gate = {}
    for init,pred in [('he',1.),('xavier',.5),('gaussian',.0512)]:
        chi = runs[f'E1_{init}_96']['initial']['chi']
        e1_gate[init] = {**chi,'predicted':pred,'relative_fwd_error':abs(chi['fwd']/pred-1),
                        'relative_bwd_error':abs(chi['bwd']/pred-1)}
    growth = {}
    thresholds = {}
    for arch in ARCHS:
        failed = []
        for depth in DEPTHS:
            label = f'E2_{arch}_{depth}'
            train = runs[label]['summary']['trainability']
            if train['numerical_failure'] or (train['rho_T'] is not None and train['rho_T']>=.9):
                failed.append(depth)
            if arch != 'plain':
                p = runs[label]['initial']
                c,q = p['c'],p['q_ratio']
                unit = arch=='residual_unit'
                predicted = (1+c if unit else 1+c/depth)**depth
                growth[label] = {'c':c,'q_ratio':q,'growth_per_block':p['growth_per_block'],
                     'predicted_ratio':predicted,'relative_error_per_block':abs(p['growth_per_block']/(1+c)-1) if unit else None,
                     'scaled_bound':1.2*math.exp(c) if not unit else None,
                     'pass':abs(p['growth_per_block']/(1+c)-1)<=.2 if unit else q<=1.2*math.exp(c)}
        thresholds[arch] = min(failed) if failed else None
    # CF5 is explicitly exploratory: selection on E2 current-depth metrics only.
    candidates = []
    eligible = [f'E2_{ar}_{d}' for ar in ARCHS for d in DEPTHS[:-1]]
    for l1,l2 in itertools.combinations(eligible,2):
        v1,v2 = [runs[l]['summary'].get('validation_metric') for l in [l1,l2]]
        g1,g2 = met(l1,'GI'),met(l2,'GI')
        if None not in [v1,v2,g1,g2] and abs(v1-v2)<=.02 and abs(g1-g2)>=.5:
            same_depth = l1.rsplit('_',1)[1] == l2.rsplit('_',1)[1]
            candidates.append((same_depth,abs(g1-g2),l1,l2,abs(v1-v2)))
    pair = max(candidates) if candidates else None
    if pair is None:
        raise RuntimeError('CF5 has no qualifying observed E2 pair (val gap<=.02, GI gap>=.5); add a documented diagnostic experiment')
    _,_,label1,label2,accuracy_gap = pair
    current_labels = [label1,label2]
    next_labels = [l.rsplit('_',1)[0]+'_'+str(DEPTHS[DEPTHS.index(int(l.rsplit('_',1)[1]))+1]) for l in current_labels]
    cf5_labels = current_labels+next_labels
    measured_chi = e1_gate['he']['fwd']
    predicted256 = measured_chi**255
    widened = runs['CF2_gaussian_width1024_depth96']['initial']['chi']
    diagnostic = runs['E2_scaled96_growth_diagnosis']['initial']
    branch_rows = list(diagnostic['branch_moments'].values())
    cs = [r['c'] for r in branch_rows]
    diagnostic_numbers = {'first_c':cs[0],'mean_c':statistics.mean(cs),'min_c':min(cs),'max_c':max(cs),
        'q_ratio':diagnostic['q_ratio'],'exact_product':math.prod(r['predicted_local_ratio_with_cross'] for r in branch_rows),
        'product_without_cross':math.prod(1+r['c']/96 for r in branch_rows),
        'registered_bound':1.2*math.exp(cs[0]),'violation_fraction':diagnostic['q_ratio']/(1.2*math.exp(cs[0]))-1}
    # Every derived cell retains an exact, resolvable source label and RUN_ID.
    write(ROOT/'results.json', {'selected_lr':lr,'runs':runs,'E1_thresholds':e1_gate,'growth':growth,
          'first_untrainable_depth':thresholds,'E3':{'A':a,'B':b,'deltas':deltas,'mean':mean,'sample_std':sd,'established':established},
          'growth_diagnosis':diagnostic_numbers,
          'CF1_q256_over_q1':predicted256,'CF2':widened,'CF5':{'labels':cf5_labels,'selection':'exploratory E2, val gap<=.02 and GI gap>=.5, prefer equal current depth','accuracy_gap':accuracy_gap},
          'runtime_seconds':sum(r['elapsed_seconds_including_setup'] for r in ledger)})
    # E1: every depth and every initialization, all three monitored quantities.
    fig,axes = plt.subplots(4,3,figsize=(9,12),constrained_layout=True)
    for i,depth in enumerate(DEPTHS):
        for init in ['he','xavier','gaussian']:
            r = runs[f'E1_{init}_{depth}']['initial']
            xs = list(range(1,depth+1))
            values = [[v['q'] for v in r['activations'].values()],
                      [v['delta']['q'] for v in r['activations'].values()],
                      [r['gradients'][f'layers.{j}.linear']['g'] for j in range(depth)]]
            for j,ys in enumerate(values):
                axes[i,j].semilogy(xs,ys,label=init)
                axes[i,j].grid(alpha=.25)
                axes[i,j].set_xlabel('Hidden layer')
                axes[i,j].set_title(f'L={depth}: '+[r'$q_l$',r'$\langle\delta_l^2\rangle$',r'$g_l$'][j])
        axes[i,0].legend(fontsize=11)
    fig.savefig(ROOT/'images/e1.png',dpi=200);plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for arch in ARCHS:
        for ax,key in zip(axes.flat,['rho_T','GI','AVD','validation_metric']):
            ys = [runs[f'E2_{arch}_{d}']['summary'].get(key) if key=='validation_metric' else met(f'E2_{arch}_{d}',key) for d in DEPTHS]
            ys = [None if runs[f'E2_{arch}_{d}']['summary']['trainability']['numerical_failure'] else y for d,y in zip(DEPTHS,ys)]
            ax.plot(DEPTHS,[np.nan if y is None else y for y in ys],marker='o',label=NAMES[arch])
            ax.set(xlabel='Depth (hidden layers / residual blocks)',ylabel=key)
            ax.grid(alpha=.25)
    axes[0,0].axhline(.5,color='green',ls=':');axes[0,0].axhline(.9,color='red',ls=':')
    axes[0,0].legend(fontsize=10)
    fig.savefig(ROOT/'images/e2.png',dpi=200);plt.close(fig)
    matrix = np.array([[met(f'E2_{ar}_{d}','rho_T') if met(f'E2_{ar}_{d}','rho_T') is not None else np.nan for d in DEPTHS] for ar in ARCHS])
    fig,ax = plt.subplots(figsize=(9,3.4),constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(matrix),cmap='viridis_r',vmin=0,vmax=1.05,aspect='auto')
    for i in range(3):
        for j in range(4):
            ax.text(j,i,'NUMERICAL\nFAILURE' if np.isnan(matrix[i,j]) else f'{matrix[i,j]:.3f}',ha='center',va='center',color='black',fontsize=12)
    ax.set(xticks=range(4),xticklabels=DEPTHS,yticks=range(3),yticklabels=[NAMES[a] for a in ARCHS],xlabel='Depth L')
    fig.colorbar(im,ax=ax,label='rho_T; fixed training subset')
    fig.savefig(ROOT/'images/map.png',dpi=200);plt.close(fig)
    cp = runs['CF3_weight_bias_decomposition']['initial']
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),constrained_layout=True)
    xs=list(range(1,25))
    for kind in ['weight','bias']:
        axes[0].semilogy(xs,[cp['gradients'][f'layers.{i}.linear']['per_parameter'][kind]['g'] for i in range(24)],label=kind)
    axes[0].semilogy(xs,[cp['gradients'][f'layers.{i}.linear']['g'] for i in range(24)],ls=':',label='combined g')
    acts=list(cp['activations'].values())
    axes[1].semilogy(xs[1:],[v['q']**.5 for v in acts[:-1]],label='RMS h(l-1)')
    axes[1].semilogy(xs[1:],[v['delta']['q']**.5 for v in acts[1:]],label='RMS delta(l)')
    for ax in axes:
        ax.set_xlabel('Hidden layer');ax.grid(alpha=.25);ax.legend(fontsize=10)
    axes[0].set_ylabel('Normalized parameter gradient');axes[1].set_ylabel('Signal / output gradient RMS')
    fig.savefig(ROOT/'images/cf3.png',dpi=200);plt.close(fig)
    # Report is six sections exactly, with registry within section 6.
    text = [r'''\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}
\usepackage[russian]{babel}
\usepackage[margin=20mm]{geometry}
\usepackage{amsmath,booktabs,graphicx,longtable,array}
\usepackage[unicode,hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\emergencystretch=2em
\begin{document}
\begin{center}{\Large\bfseries Лабораторная работа 1}\\[4pt]
Почему глубокая сеть обучается или не обучается\\
Ван Чжаофу · Томский политехнический университет · P3adf02
\end{center}
\section{Профиль и протокол}
Методические указания: редакция 4 от 08.10.2026, все 13 страниц проверены. Глубины 8,24,48,96; ширина 256; $\sigma=0.02$; SGD с импульсом 0.9. Применяется разрешённый в §6 CIFAR-10. Механизм экспериментов, данные, контрольные точки и воспроизводимость сохранены из ЛР0.
Полная обучающая роль: 43000 изображений; val 5000; cal 2000; test 10000. Выбор условий не использует test. Нормализация вычислена только по train. В ЛР1 используется вся обучающая роль, а не 25\%, использованные в ЛР0.
Протокол данных преподавателя не предоставлен. Используется явное сохранённое разбиение ЛР0: seed 1729, по 500 объектов каждого класса для val и по 200 для cal. Не утверждается совпадение индексов с отсутствующим файлом преподавателя. Роль cal зарезервирована и не используется.
Три полные эпохи; batch 128; weight decay 0; постоянная скорость обучения; без AMP и ограничения градиента. Для завершённых трёхэпоховых обучений сохранены 1008 шагов; для одноэпоховой проверки Gaussian -- 336. Численные сбои остановлены раньше и зарегистрированы отдельно. Архитектура, seed, хеши кода/данных и точный RUN\_ID находятся в паспорте. Обозначения Rxx ниже разрешаются в полные RUN\_ID в реестре в конце отчёта.
Обычная сеть имеет $L$ скрытых линейных слоёв. Остаточная сеть содержит входную проекцию 3072$\to$256 и $L$ блоков, по два линейных слоя в каждом; после сложения нет активации. Поэтому число параметров и линейных преобразований различается; сравнение архитектур не является сравнением равных бюджетов параметров.
Проба: первые 64 фиксированных объекта train, режим eval, $\|u\|=1$, генератор $u$ из seed профиля 987693601. Для измерений используется float64-копия исходных float32-весов. Это исключает потерю очень малых сигналов при диагностике; обучение остаётся float32. Копирование не меняет веса, режим, BN-буферы, RNG и градиенты обучаемой модели.
Окончательная $\rho_T=L_T/\ln10$ измеряется по первым 1024 фиксированным объектам train после обучения. GI вычисляется по всем непосредственно параметризованным слоям, включая проекцию и классификатор: сначала медиана $g_l=\|\nabla_{\theta_l}L\|/\sqrt{|\theta_l|}$ по первому, среднему и последнему batch каждой эпохи, затем $\log_{10}[(\max g_l+10^{-12})/(\min g_l+10^{-12})]$. Вес и bias входят в один слой. AVD вычисляется по дисперсиям скрытых выходов фиксированной eval-пробы, включая $h_0$ остаточной сети. Максимальная доля нулей берётся на настоящих ReLU-выходах, а не на сумме ветви и skip.
Монитор также сохраняет среднее, std, min/max, второй момент, насыщение tanh при $|1-h^2|<0.1$, нормы и относительные нормы градиента. Квадраты вычисляются после приведения к float64. Отсутствующий градиент классификатора в линейной пробе отмечен как unused, а не как нулевой измеренный градиент.
''']
    text.append('Скорость обучения выбрана один раз по максимальной итоговой val accuracy; при равенстве используется меньшая val loss, затем меньшая LR. Все кандидаты обучались три эпохи. Выбрана LR='+number(lr)+'.')
    text.append('Код поисковой серии: '+runs['E1_he_8']['manifest']['git_commit'][:7]+'; код E3 и дополнительных разложений: '+runs['E3_A_seed1']['manifest']['git_commit'][:7]+'. Между сериями добавлены разложение градиента по weight/bias, запись моментов всех ветвей и защита сериализации нечисловых состояний; модель и правила конечного обучения не изменены. Обе версии исходников проверяются по Git и сохранённым хешам.')
    text.append(table(['LR','val accuracy','val loss','Источник'],[[number(float(k[3:])),number(r['summary'].get('validation_metric')),number(r['summary'].get('validation_loss')),r['ref']] for k,r in runs.items() if k.startswith('LR_')]))
    budget = [r for k,r in runs.items() if k.startswith('budget_')]
    text.append('Предварительная проверка времени не заменяет формальные серии: '+ '; '.join(r['ref']+': '+number(r['summary'].get('epoch_time'))+' с/эпоха' for r in budget)+'.')
    text.append(r'\section{E1: сигнал и градиент при инициализации}')
    text.append(r'Заранее записанная гипотеза: по (1.4)--(1.8) ожидаются $\chi=1$ (He), $0.5$ (Xavier), $n\sigma^2/2=0.0512$ (Gaussian). Выполнены все 12 комбинаций без оптимизатора. Множители оцениваются геометрически между равными по ширине скрытыми слоями; при $q<10^{-30}$ применяется правило исключения из PDF. Для Gaussian L=96 эффективная конечная точка может быть раньше последнего слоя; это не замена полного обратного прохода.')
    e1_rows = []
    for init,pred in [('he',1),('xavier',.5),('gaussian',.0512)]:
        for d in DEPTHS:
            label=f'E1_{init}_{d}';c=runs[label]['initial']['chi']
            e1_rows.append([init,d,number(pred),number(c['fwd']),number(c['bwd']),c['effective_layers'],ref(label)])
    text.append(table(['Init','L',r'$\chi$',r'$\hat\chi_f$',r'$\hat\chi_b$',r'$L_{eff}$','Источник'],e1_rows))
    text.append('На максимальной глубине относительные ошибки (forward; backward): '+ '; '.join(init+': '+number(v['relative_fwd_error']*100)+r'\%; '+number(v['relative_bwd_error']*100)+r'\%' for init,v in e1_gate.items())+'. Все шесть сравнений проверяются с допуском 20\%.')
    text.append('Граница применимости: Gaussian после одной полной эпохи с выбранной общей LR. Результаты повторной пробы:')
    text.append(table(['L',r'$\hat\chi_f$',r'$\hat\chi_b$','Учтено / посл. слой',r'$\rho_T$','Источник'],[[d,number(runs[f'E1_after_epoch_gaussian_{d}']['final']['chi']['fwd']),number(runs[f'E1_after_epoch_gaussian_{d}']['final']['chi']['bwd']),str(runs[f'E1_after_epoch_gaussian_{d}']['final']['chi']['effective_layers'])+' / '+runs[f'E1_after_epoch_gaussian_{d}']['final']['chi']['last_layer'][1:],number(met(f'E1_after_epoch_gaussian_{d}','rho_T')),ref(f'E1_after_epoch_gaussian_{d}')] for d in DEPTHS]))
    dead=runs['E1_after_epoch_gaussian_48']['final']['activations']
    text.append('Прочерк в backward для '+ref('E1_after_epoch_gaussian_48')+' означает неопределённую оценку: q последнего скрытого выхода='+number(dead['h48']['q'])+', средний квадрат delta первого слоя='+number(dead['h1']['delta']['q'])+'. Последний ReLU равен нулю на всей фиксированной пробе, и все предыдущие delta равны нулю. Отношение 0/0 не заменяется нулём или теоретическим множителем. Все эти значения конечны; флаг NaN/Inf поэтому не ставится.')
    hidden_medians=[v for k,v in runs['E1_after_epoch_gaussian_96']['summary']['trainability']['median_g_by_layer'].items() if k.startswith('layers.')]
    text.append('Для '+ref('E1_after_epoch_gaussian_96')+' максимальная медиана g скрытых слоёв при реальном float32-обучении равна '+number(max(hidden_medians))+'. Ненулевой градиент float64-пробы не означает ненулевого градиента при float32-обучении: численное обнуление путей и статистика в повышенной точности должны различаться.')
    text.append('Уже первое обновление SGD связывает веса с обучающими входами через градиент и в общем случае меняет нулевые bias. Поэтому исходные независимые случайные веса и нулевые смещения (1.4) больше не предполагаются. Эти нарушения возникают в одном шаге оптимизатора; измерение после эпохи не позволяет приписать одному из них более ранний момент. Совпадение части множителей с теорией не восстанавливает предположения; изменение bias способно создать сигнал даже при затухающих весовых путях.')
    biases=read(ROOT/'bias_audit.json')
    text.append('Проверка фактических checkpoint: максимальное абсолютное bias после эпохи '+ '; '.join(ref(r['label'])+': '+number(r['final_max_abs_bias']) for r in biases)+'. Начальное bias во всех случаях равно нулю; агрегат включает классификатор и скрытые слои.')
    text.append(r'\clearpage')
    text.append(figure('e1','Сигнал, градиент по выходу и нормированный градиент параметров: все глубины и три схемы. Источники: '+', '.join(ref(f'E1_{init}_{d}') for d in DEPTHS for init in ['he','xavier','gaussian'])+'.','.88'))
    text.append(r'\clearpage')
    text.append(r'\section{E2: глубина и остаточные связи}')
    text.append(r'Заранее записанная гипотеза: He сохраняет масштаб обычной сети лишь в среднем; для остаточной сети (1.13) предсказывает $(1+\alpha^2c)^L$, $c\approx2$. При $\alpha=1$ сигнал растёт экспоненциально, а при $\alpha=L^{-1/2}$ ограничивается $e^c$. Оптимизация и конечная ширина могут нарушать независимость слагаемых.')
    rows=[]
    for arch in ARCHS:
        for d in DEPTHS:
            label=f'E2_{arch}_{d}';r=runs[label];t=r['summary']['trainability']
            rows.append([NAMES[arch],d,number(t['rho_T']),number(t['GI']),number(t['AVD']),number(r['summary'].get('validation_metric')),'сбой' if t['numerical_failure'] else 'OK',r['ref']])
    text.append(table(['Архитектура','L',r'$\rho_T$','GI','AVD','val acc.','Статус','Источник'],rows))
    text.append('Численный сбой является наблюдаемым отрицательным результатом, а не завершёнными тремя эпохами. Для него отсутствующие итоговые метрики обозначены прочерком; не используются случайная точность или нулевая ошибка вместо отсутствующего измерения.')
    text.append('GI при сбое вычислен только по доступным измерениям до остановки, поэтому не сопоставляется как полноценная девятиточечная медиана с завершёнными запусками. Доли нулей в сводке сбоя относятся к исходной пробе. Точная остановка:')
    text.append(table(['L','Завершено эпох','Шагов SGD','Источник'],[[d,runs[f'E2_residual_unit_{d}']['manifest']['completed_epoch'],runs[f'E2_residual_unit_{d}']['manifest']['global_step'],ref(f'E2_residual_unit_{d}')] for d in DEPTHS]))
    text.append('Первая необучаемая глубина (rho≥0.9 либо численный сбой): '+ '; '.join(NAMES[a]+': '+('не найдена в профиле' if thresholds[a] is None else str(thresholds[a])) for a in ARCHS)+'.')
    text.append(table(['Архитектура','L','Параметры','Линейных слоёв','Источник'],[[NAMES[a],d,runs[f'E2_{a}_{d}']['manifest']['parameters'],d+1 if a=='plain' else 2*d+2,ref(f'E2_{a}_{d}')] for a in ARCHS for d in [8,96]]))
    text.append(figure('e2','Результаты трёх эпох: численно неуспешные условия не выводятся как итоговые кривые. Частичные GI до сбоя сохранены в таблице. Источники соответствуют таблице E2.','.47'))
    text.append('Проверка роста сигнала до обучения. Для единичной ветви сравнивается корень степени L, поскольку прямое отношение накапливает малые ошибки каждого блока. Для масштабированной ветви применяется именно верхняя граница 1.2 exp(c).')
    text.append(table(['Ветвь','L','c',r'$q_L/q_0$',r'$(q_L/q_0)^{1/L}$','Порог','Источник'],[[('1' if 'unit' in label else r'$L^{-1/2}$'),int(label.rsplit('_',1)[1]),number(g['c']),number(g['q_ratio']),number(g['growth_per_block']),('PASS' if g['pass'] else 'FAIL'),ref(label)] for label,g in growth.items()]))
    text.append(r'\textbf{Зарегистрированное исключение:} на L=96 масштабированная ветвь превышает границу на '+number(100*diagnostic_numbers['violation_fraction'])+r'\%. Этот численный порог не пройден; его нельзя объявить PASS по результатам обучения или подменить другим seed.')
    text.append('Отдельная проба с теми же весами и batch ('+ref('E2_scaled96_growth_diagnosis')+') локализует причину: c первого блока '+number(cs[0])+', среднее c по96 блокам '+number(statistics.mean(cs))+', диапазон ['+number(min(cs))+', '+number(max(cs))+']. Одно измерение первого блока не является общей константой всех ветвей конечной сети.')
    text.append(r'Точное равенство для измеренных моментов: $q_{l+1}/q_l=1+\alpha^2c_l+2\alpha\kappa_l$, где $\kappa_l=\langle h_l,F_l\rangle/q_l$. Произведение этих множителей равно '+number(diagnostic_numbers['exact_product'])+', непосредственно измеренное отношение '+number(diagnostic_numbers['q_ratio'])+'. Если сохранить все разные $c_l$, но отбросить перекрёстные члены, получится '+number(diagnostic_numbers['product_without_cross'])+'. Значит, в данном запуске перекрёстные члены суммарно уменьшают рост; превышение границы связано прежде всего с использованием заниженного c первого блока вместо разных $c_l$. Предположения общего c и нулевого эмпирического перекрёстного момента не выполняются точно. Равенство с полными моментами проверено отдельным автоматическим тестом. Исходный FAIL остаётся в таблице и acceptance.json.')
    text.append(r'\section{Карта обучаемости}')
    text.append(figure('map',r'Двумерный срез: глубина $\times$ архитектура $\to\rho_T$. Численный сбой показан текстом, а не искусственным значением.','.35'))
    text.append('Изменяются только глубина и архитектура; фиксированы: CIFAR-10 и его разбиение, полный train, нормализация данных, seed профиля, ширина256, He, ReLU, отсутствие нормализации слоёв, SGD momentum0.9, LR='+number(lr)+', batch128, три эпохи, float32, отсутствие AMP и клиппинга, подмножество вычисления rho и расписание измерений. Источники ячеек: таблица E2 и реестр RUN\_ID.')
    text.append(r'Порог $\rho_T\le0.5$ обозначает обучаемость; $\rho_T\ge0.9$ -- необучаемость; промежуточные значения -- пограничный режим. Более высокая val accuracy сама по себе не означает выполнения порога обучаемости. Карта относится к одному seed и одному разбиению и не доказывает универсальную границу глубины.')
    text.append(r'\section{E3: основное парное сравнение}')
    text.append('Гипотеза и точная конфигурация сохранены в hypothesis.md и experiments/lab1/e3\_preregistration.json до всех запусков E3. Доказательство Git-порядка: preregistration\_evidence.json. A: '+tex(prereg['description_A'])+'. B: '+tex(prereg['description_B'])+'. Единственный изменяемый фактор: '+tex(prereg['factor'])+'.')
    text.append(r'Основная метрика: $\rho_T$, меньше лучше. Ожидается положительная разность A−B. Условие опровержения: отсутствие трёх положительных разностей либо средняя разность меньше двух выборочных стандартных отклонений. Все остальные условия совпадают, включая буквальные seeds1,2,3.')
    text.append(table(['seed',r'$\rho_T(A)$',r'$\rho_T(B)$',r'$\Delta_s$','A / B'],[[s,number(a[s-1]),number(b[s-1]),number(deltas[s-1]),ref(f'E3_A_seed{s}')+' / '+ref(f'E3_B_seed{s}')] for s in [1,2,3]]))
    text.append('Средняя разность '+number(mean)+', выборочное sΔ='+number(sd)+', 2sΔ='+number(2*sd)+'. По установленному правилу: '+('различие установлено.' if established else 'различие не установлено.')+' Это проверка главного эффекта на данном разбиении; три seeds отражают только случайность обучения, а не перенос на другие данные.')
    text.append(table(['Вариант','Среднее rho','SD rho','Среднее val acc.','SD val acc.'],[[arm,number(statistics.mean(vals)),number(statistics.stdev(vals)),number(statistics.mean([runs[f'E3_{arm}_seed{s}']['summary']['validation_metric'] for s in [1,2,3]])),number(statistics.stdev([runs[f'E3_{arm}_seed{s}']['summary']['validation_metric'] for s in [1,2,3]]))] for arm,vals in [('A',a),('B',b)]]))
    text.append(r'\section{Контрфактические вопросы и проверка результатов}')
    text.append(r'\subsection*{1. Экстраполяция He до глубины 256}')
    text.append(r'Из '+ref('E1_he_96')+r': $\hat\chi_f='+number(measured_chi)+r'$. Формальная экстраполяция даёт $q_{256}/q_1=\hat\chi_f^{255}='+number(predicted256)+r'$. Это прогноз по (1.4), а не измерение L=256. Даже небольшое отклонение множителя от единицы накапливается экспоненциально. Стабильность на L=16 не гарантирует ни сохранения масштаба, ни оптимизации на L=256: конечная ширина создаёт корреляции активаций и производных; при обратном проходе предположение независимости градиента от тех же весов уже приближённо. После первого обновления нарушается и предположение исходных независимых весов. Скалярный второй момент не контролирует весь спектр Якобиана.')
    text.append(r'\subsection*{2. Увеличение ширины в четыре раза}')
    text.append(r'При n=1024 и прежнем $\sigma=0.02$ прогноз $\chi=1024\sigma^2/2=0.2048$. Отдельная проба '+ref('CF2_gaussian_width1024_depth96')+' дала forward '+number(widened['fwd'])+', backward '+number(widened['bwd'])+', эффективных слоёв '+str(widened['effective_layers'])+'. Рост ширины изменяет множитель, но не делает его единичным; sigma не перенастраивалась.')
    text.append(r'\subsection*{3. Градиент по выходу и градиент по весу}')
    gprobe = runs['E1_gaussian_24']['initial']
    grads = [gprobe['gradients'][f'layers.{j}.linear']['g'] for j in range(24)]
    ds = list(gprobe['activations'].values())
    text.append('Источник '+ref('E1_gaussian_24')+': средние квадраты delta на первом и последнем скрытых слоях '+number(ds[0]['delta']['q'])+' и '+number(ds[-1]['delta']['q'])+'; g первого и последнего скрытых слоёв '+number(grads[0])+' и '+number(grads[-1])+'.')
    text.append(r'Для весов $\nabla_{W_l}L$ содержит произведение входа $h_{l-1}$ и обратного сигнала с маской ReLU. По (1.4), (1.8) уменьшение прямого сигнала при продвижении к выходу компенсируется ростом delta в том же направлении; поэтому весовые градиенты внутренних слоёв имеют близкий порядок. Первый слой имеет другую ширину входа и статистику. Однако в обязательном $g_l$ здесь учтены также bias: $\nabla_bL$ не умножается на малый вход. Поэтому полное g может существенно расти к выходу, и буквально одинаковые значения на исходном графике не заявляются. Этот предел подсказки PDF и отличие bias от весов проверяются отдельным сохранённым разложением, если оно включено в реестр. Константа epsilon в GI дополнительно маскирует различия величин намного меньше $10^{-12}$; такой GI не означает сохранения абсолютных градиентов.')
    if 'CF3_weight_bias_decomposition' in runs:
        cp = runs['CF3_weight_bias_decomposition']['initial']['gradients']
        ws=[cp[f'layers.{i}.linear']['per_parameter']['weight']['g'] for i in range(1,24)]
        bs=[cp[f'layers.{i}.linear']['per_parameter']['bias']['g'] for i in range(1,24)]
        text.append('Отдельная проба '+ref('CF3_weight_bias_decomposition')+': внутренние весовые g (слои2–24) min='+number(min(ws))+', max='+number(max(ws))+', max/min='+number(max(ws)/min(ws))+'; bias g min='+number(min(bs))+', max='+number(max(bs))+'. Это измеренное разложение, а не изменение определения обязательного GI.')
        text.append(figure('cf3','Разложение весов и bias; компенсация масштабов прямого и обратного сигналов. Источник '+ref('CF3_weight_bias_decomposition')+'.','.32'))
    text.append(r'\subsection*{4. Почему единичный skip не предотвращает взрыв}')
    text.append(table(['L','Измерено q ratio',r'Прогноз $(1+c)^L$','Ошибка корня','Источник'],[[d,number(growth[f'E2_residual_unit_{d}']['q_ratio']),number(growth[f'E2_residual_unit_{d}']['predicted_ratio']),number(growth[f'E2_residual_unit_{d}']['relative_error_per_block']),ref(f'E2_residual_unit_{d}')] for d in DEPTHS]))
    text.append(r'Единичный член Якобиана обеспечивает прямой путь, но не убирает вклад ветви. При $\alpha=1$ накопление энергии ветвей экспоненциально; огромные логиты и производные делают обучение неустойчивым. Формула предсказывает средний рост при инициализации, а не устойчивость конечного шага SGD. Расхождения полных отношений при глубине96 могут быть большими даже при допустимой ошибке среднего множителя за блок.')
    text.append(r'\subsection*{5. Близкая точность не означает одинаковую устойчивость}')
    text.append('Пара выбрана исследовательски из E2: разность val accuracy не более0.02, разность GI не менее0.5; следующая глубина для каждой конфигурации уже содержится в сетке E2, поэтому это проверка на имеющихся данных, а не независимый прогноз. Текущие конфигурации: '+ref(label1)+' и '+ref(label2)+'; проверка следующей глубины: '+', '.join(ref(l) for l in next_labels)+'. Глубины исходной пары могут различаться; следующий уровень берётся отдельно для каждой архитектуры.')
    text.append(table(['Архитектура','L','val acc.','GI',r'$\rho_T$','Источник'],[[NAMES[l.split('_',1)[1].rsplit('_',1)[0]],int(l.rsplit('_',1)[1]),number(runs[l]['summary'].get('validation_metric')),number(met(l,'GI')),number(met(l,'rho_T')),ref(l)] for l in cf5_labels]))
    changes=[]
    for current,following in zip(current_labels,next_labels):
        changes.append(ref(current)+'→'+ref(following)+': изменение val accuracy '+number(runs[following]['summary']['validation_metric']-runs[current]['summary']['validation_metric'])+', изменение rho '+number(met(following,'rho_T')-met(current,'rho_T')))
    text.append('Фактические изменения при увеличении глубины: '+'; '.join(changes)+'. Одинаково устойчивыми эти конфигурации считать нельзя. В выбранной паре масштабированная остаточная сеть сохраняет точность лучше, хотя её GI выше: GI не является достаточным сертификатом устойчивости при сравнении разных архитектур, масштаба ветви и нормировки числа параметров. Проверка соседней глубины не даёт права объявить общий причинный закон по четырём точкам.')
    text.append(r'\subsection*{Воспроизводимость и полнота}')
    import xml.etree.ElementTree as ET
    tests=ET.parse('docs/environment/tests-lab1-final.xml').getroot().find('testsuite').attrib
    text.append('Итоговая автоматическая проверка: '+tests['tests']+' тестов; failures='+tests['failures']+', errors='+tests['errors']+', skipped='+tests['skipped']+'. Включены все пять обязательных видов проверок и регрессия ЛР0. Источник: docs/environment/tests-lab1-final.xml.')
    text.append('Все результаты, графики и таблицы получены программой reports/tools/summarize\_lab1.py из сохранённых JSON/JSONL. Обязательные пять видов тестов и регрессия ЛР0 отражены в JUnit XML; численные сбои сохраняют failure.json и не заменяются smoke-тестами. Дополнительные задания §14 (отдельные серии нормализации, оптимизаторов и SVD) не заявлены выполненными. Ограничения: CIFAR-10 вместо основного FashionMNIST разрешён PDF; одна фиксированная выборка; лишь три эпохи; разные параметрические бюджеты plain/residual; диагностика float64 не означает float64-обучение.')
    text.append('Суммарное время сохранённых запусков с подготовкой: '+number(sum(r['elapsed_seconds_including_setup'] for r in ledger)/60)+' мин; отдельно учитываются тесты, разработка и компиляция. Точные времена и статусы каждого запуска находятся в results.json и ledger.json.')
    text.append(r'\subsection*{Реестр источников чисел}')
    text.append(r'{\scriptsize\begin{longtable}{p{7mm}p{43mm}p{103mm}}\toprule Код & Эксперимент & RUN\_ID\\\midrule\endhead')
    for label,r in runs.items():
        text.append(r['ref']+' & '+tex(label)+r' & {\ttfamily '+r['RUN_ID']+r'} \\')
    text.append(r'\bottomrule\end{longtable}}')
    text.append(r'\end{document}')
    content='\n\n'.join(text)+'\n'
    for old,new in [('8,24,48,96','8, 24, 48, 96'),('seeds1,2,3','seeds 1, 2, 3'),('ширина256','ширина 256'),
                    ('momentum0.9','momentum 0.9'),('batch128','batch 128'),('по96','по 96'),('глубине96','глубине 96')]:
        content=content.replace(old,new)
    # Separate the protocol's conceptual paragraphs instead of one dense text block.
    content=content.replace('сохранены из ЛР0.\n','сохранены из ЛР0.\n\n')
    for start in ['Полная обучающая роль:','Протокол данных преподавателя','Три полные эпохи;','Обычная сеть имеет',
                  'Проба: первые','Окончательная $','Монитор также']:
        content=content.replace('\n'+start,'\n\n'+start)
    content=content.replace(r'программой reports/tools/summarize\_lab1.py',r'программой \path{reports/tools/summarize_lab1.py}')
    (ROOT/'main.tex').write_text(content,encoding='utf-8')
    # Evidence acceptance distinguishes observations, numerical thresholds and completion.
    formal = [r for r in runs.values() if r['manifest']['purpose']=='formal']
    gates = {'E1_12_runs':all(f'E1_{i}_{d}' in runs for i in ['he','xavier','gaussian'] for d in DEPTHS),
             'E1_chi_within_20_percent':all(v['relative_fwd_error']<=.2 and v['relative_bwd_error']<=.2 for v in e1_gate.values()),
             'E1_after_epoch_all_depths':all(f'E1_after_epoch_gaussian_{d}' in runs for d in DEPTHS),
             'E2_12_runs':all(f'E2_{a}_{d}' in runs for a in ARCHS for d in DEPTHS),
             'E2_growth_thresholds':all(g['pass'] for g in growth.values()),
             'E2_growth_failure_diagnosed':abs(diagnostic_numbers['exact_product']/diagnostic_numbers['q_ratio']-1)<1e-10 and abs(diagnostic['q_ratio']/runs['E2_residual_scaled_96']['initial']['q_ratio']-1)<1e-12,
             'E3_six_completed_runs':all(runs[f'E3_{a}_seed{s}']['manifest']['status']=='COMPLETED' for a in ['A','B'] for s in [1,2,3]),
             'formal_clean_git':all(not r['manifest']['git_dirty'] for r in formal),
             'all_formal_real_data':all(not r['manifest']['synthetic_fixture'] for r in formal),
             'CF5_observed_pair':pair is not None,'six_report_sections':'\n\\section{' in '\n'.join(text) and sum(t.startswith('\\section{') for t in text)==5}
    # First section appears in the TeX preamble string above; total sections is six.
    gates['six_report_sections'] = (ROOT/'main.tex').read_text(encoding='utf-8').count('\\section{')==6
    write(ROOT/'acceptance.json', {'gates':gates,'numeric_thresholds_pass':all(gates.values()),
           'E3_difference_established':established,'E1_details':e1_gate,'E2_growth':growth,
           'first_untrainable_depth':thresholds,'pending':['final test XML','preregistration Git ancestry and timestamp audit','compiled PDF all-page visual QA','Lab0 immutable file audit']})
    print(json.dumps({'E3_deltas':deltas,'E3_mean':mean,'E3_std':sd,'established':established,'CF5':cf5_labels,'gates':gates},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
