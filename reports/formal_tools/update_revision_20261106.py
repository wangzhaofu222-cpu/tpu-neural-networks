"""Append real supplementary evidence and update the two affected formal reports."""
from common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REV=ROOT/'experiments/revision_20261106'
BACK=REPORTS/'revision_20261106';BACK.mkdir(exist_ok=True)

def original_body(n):
    out=REPORTS/f'lab{n}'/'formal';p=BACK/f'lab{n}_report_body_before.tex'
    if not p.exists():
        shutil.copy2(out/'report_body.tex',p)
        shutil.copy2(out/f'Lab{n}_Report.pdf',BACK/f'Lab{n}_Report_before.pdf')
    return out,p.read_text('utf-8')

def add_registry_rows(body,rows):
    text=''
    for alias,condition,rid in rows:
        text+='\\hypertarget{'+alias+'}{'+alias+'} & '+condition+' & {\\fontsize{8.8}{11}\\selectfont\\texttt{'+rid+'}} \\\\\n'
    mark='\\bottomrule\\end{longtable}\\endgroup'
    assert body.count(mark)==1
    return body.replace(mark,text+mark)

def lab1():
    out,body=original_body(1);d=load(REV/'lab1/results.json');rows=[];registry=[]
    fmt=lambda v:'---' if v is None else f'{v:.6f}'
    for i,r in enumerate(d['rows'],1):
        alias='S'+str(i);label=r['label'];arch='Plain' if 'plain' in label else r'Res. $\alpha=1$' if 'unit' in label else r'Res. $\alpha=L^{-1/2}$'
        tr=r['trainability'] or {}
        rows.append([arch,label.rsplit('_',1)[1],r['global_step'],fmt(tr.get('rho_T')),fmt(r.get('validation_metric')),
                     'завершён' if r['status']=='COMPLETED' else 'сбой',alias])
        registry.append((alias,'E2 repeat '+label.replace('E2_','').replace('_',' '),r['repeat_RUN_ID']))
    tab=table(out,'repeat_e2','Повтор E2: исходные seed, данные и гиперпараметры; сбои не заменены успешными запусками.',
              ['Архитектура','L','Шаги',r'$\rho_T$','Val acc.','Статус','Ист.'],rows,'lrrrrll','footnotesize')
    scaled=[r for r in d['rows'] if 'scaled' in r['label']]
    growth=table(out,'repeat_growth','Повторная проверка границы роста масштабированной остаточной сети.',
                 ['L',r'$c_1$',r'$q_L/q_0$',r'$1.2e^{c_1}$','Порог'],
                 [[r['label'].rsplit('_',1)[1],fmt(r['c']),fmt(r['q_ratio']),fmt(r['scaled_bound']),'PASS' if r['threshold_pass'] else 'FAIL'] for r in scaled],'rrrrl')
    text=r'''
\subsection*{Повторная проверка E2}
Повтор проведён после проверки реализации без изменения конфигурации: seed 987693601, SGD с momentum 0.9, learning rate 0.003, batch 128, три эпохи, ширина 256, исходный split и все двенадцать условий. Первоначальные запуски сохранены; повтор имеет отдельные RUN\_ID. Проверены инициализация обоих слоёв ветви, нулевые bias, масштаб $L^{-1/2}$, hook измерения $q$ и определение $c$ по первой ветви. Ошибка реализации не обнаружена.
@@TABLE@@
@@GROWTH@@
Начальные вторые моменты повторяют исходные значения с относительной ошибкой менее $10^{-12}$. В частности, при L=96 снова получено $q_L/q_0=6.996440$ при границе 6.849322. Исходный FAIL воспроизведён, а не устранён сменой seed или порога.

Для конечной ширины точное равенство содержит перекрёстный момент:
\begin{equation}
\frac{q_l}{q_{l-1}}=1+\alpha^2 c_l+2\alpha r_l,\quad
r_l=\frac{\mathbb E[h_{l-1}F_l(h_{l-1})]}{q_{l-1}}.
\end{equation}
Произведение этих измеренных множителей совпадает с $q_L/q_0$ с относительной ошибкой менее $10^{-12}$. Замена всех $c_l$ первым $c_1$ и отбрасывание $r_l$ дают приближение среднего поля, а не тождество отдельной конечной сети. Поэтому воспроизведение подтверждает численную причину отклонения; оно не превращает невыполненный порог в PASS. Основной трёхseedовый E3 и его исходная гипотеза не менялись.
'''.replace('@@TABLE@@',tab).replace('@@GROWTH@@',growth)
    marker='\\section{Карта обучаемости}'
    assert marker in body
    body=body.replace(marker,text+'\n'+marker)
    body=add_registry_rows(body,registry)
    (out/'report_body.tex').write_text(body,'utf-8');compile_report(1)

def lab4():
    out,body=original_body(4);d=load(REV/'lab4/results.json');conds=d['conditions'];registry=[]
    fig,axes=plt.subplots(2,1,figsize=(9,4.9),gridspec_kw={'height_ratios':[1,2]})
    examples=np.load(Path(load(REV/'lab4/with_jitter_features.json')['path'])/'class_examples.npy')
    axes[0].imshow(np.concatenate(list(examples),axis=1));axes[0].set_xticks(np.arange(10)*32+15,range(10));axes[0].set_yticks([])
    axes[0].set_title('Hue-defined classes: identical class rule for both encoders')
    x=np.arange(2);axes[1].bar(x-.18,[conds[k]['validation_accuracy'] for k in conds],.36,label='Validation')
    axes[1].bar(x+.18,[conds[k]['test_accuracy'] for k in conds],.36,label='Test')
    axes[1].set_xticks(x,['With ColorJitter','Without ColorJitter']);axes[1].set_ylim(0,1);axes[1].set_ylabel('Linear-probe accuracy');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'figures/hue_control.png',dpi=180);plt.close(fig)
    rows=[]
    for i,(k,r) in enumerate(conds.items(),1):
        alias=f'C{i}'
        rows.append(['T' if k=='with_jitter' else r'$T_{-ColorJitter}$',f"{r['selected_regularization']:.3f}",f"{r['validation_accuracy']:.6f}",f"{r['test_accuracy']:.6f}",alias])
        registry.append((alias,'Q2 hue '+k.replace('_',' '),r['RUN_ID']))
        registry.append((alias+'F','Q2 hue features',r['feature_RUN_ID']))
        for j,c in enumerate(r['candidates'],1):registry.append((alias+'P'+str(j),'Q2 probe L2 '+str(c['regularization']),c['RUN_ID']))
    tab=table(out,'hue_control','Цветовой контроль: два замороженных кодировщика E2, один seed и одинаковый probe.',
              ['Кодировщик','L2','Val accuracy','Test accuracy','Ист.'],rows,'lrrrl')
    sign='выше' if d['difference_without_minus_with']>0 else 'ниже' if d['difference_without_minus_with']<0 else 'равна'
    text=r'''
\subsection*{Вопрос 2. Класс, определяемый оттенком}
Построен HueCIFAR-10: исходные train/val/test 43\,000/5\,000/10\,000 изображений сохраняют роли, cal не читается. Семантические метки отброшены. В каждой роли случайно и точно поровну назначены десять классов оттенка, независимо от содержания, с seed 4601/4602/4603. Для класса k оттенок равен $(k+0.5)/10+u$, $u\sim U[-0.015,0.015]$, насыщенность 0.8; яркость $V=0.25+0.65(0.299R+0.587G+0.114B)$. HSV переведён в RGB с округлением до uint8. Текстура яркости сохраняется, класс задаётся только оттенком.

Использованы два уже обученных 25 эпох кодировщика E2: полный T и вариант без ColorJitter. Предобучение не повторялось, кодировщики и BatchNorm заморожены. На h обучены линейные пробы: 200 эпох, SGD 0.1, momentum 0.9, cosine, seed 1. L2 выбирается из $\{0,0.001,0.01\}$ по val; признаки стандартизуются только по train. До обучения ожидалось преимущество варианта без изменения цвета.
@@TABLE@@
\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figures/hue_control.png}
\caption{Примеры классов по оттенку и точность линейных проб на цветовой задаче.}
\end{figure}
@@RESULT@@
Это прямой контроль информации об оттенке в h при данных кодировщиках. Один seed не устанавливает устойчивость разности; результат не переносится автоматически на классификацию CIFAR по предметным классам и не означает полного удаления цвета проекционной головой.
'''.replace('@@TABLE@@',tab).replace('@@RESULT@@',f"Точность без ColorJitter {sign}; разность test accuracy «без минус с» равна {d['difference_without_minus_with']:+.6f}. "+('Предсказанное направление подтверждено в этом контроле.' if d['prediction_supported'] else 'Предсказанное направление не подтвердилось; наблюдение сохранено.'))
    body=re.sub(r'\\subsection\*\{Вопрос 2\..*?(?=\\subsection\*\{Вопрос 3\.)',lambda m:text,body,flags=re.S)
    body=body.replace('Отдельный цветовой контроль не выполнен, поэтому его вывод не заявляется.','Цветовой контроль использует один seed и замороженные кодировщики; его область применимости отделена от предметных классов CIFAR.')
    body=add_registry_rows(body,registry)
    (out/'report_body.tex').write_text(body,'utf-8');compile_report(4)

if __name__=='__main__':
    import sys
    for name in ['formal_final_audit.json','formal_report_state.json','REPORT_PROGRESS.md']:
        dest=BACK/('before_'+name)
        if not dest.exists():shutil.copy2(REPORTS/name,dest)
    # Required artifact-operation marker, once before authoring the two PDF edits.
    marker=Path('C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/pdf/26.1104.12130/skills/pdf/container_tools/mark_artifact_operation_started.mjs')
    subprocess.run(['node',str(marker),'--operation-kind','edit','--expected-output-count','2','--output-format','pdf'],check=True)
    lab1();lab4()
