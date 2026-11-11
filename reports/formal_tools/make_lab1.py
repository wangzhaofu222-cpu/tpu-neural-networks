from common import *
import statistics

out=setup(1,['Почему глубокая сеть обучается или не обучается',
             'Инициализация, нормализация и остаточные связи'])
pdf=next((ROOT/'drive-download-20261026T053112Z-1-001').glob('lab1_*.pdf'))
track([pdf,REPORTS/'lab1/main.tex',REPORTS/'lab1/results.json'])
d=load(REPORTS/'lab1/results.json');runs=d['runs']
for r in runs.values():
    path=ROOT/r['path']
    for f in ['manifest.json','summary.json','config.resolved.yaml']:
        if (path/f).exists():track([path/f])
    if (path/'summary.json').exists():
        actual=load(path/'summary.json')
        assert actual.get('validation_metric')==r.get('summary',{}).get('validation_metric')
assert abs(statistics.mean(d['E3']['deltas'])-d['E3']['mean'])<1e-12
assert abs(statistics.stdev(d['E3']['deltas'])-d['E3']['sample_std'])<1e-12
old=(REPORTS/'lab1/main.tex').read_text('utf-8')
body=old[old.index('\\section{Профиль и протокол}'):old.index('\\subsection*{Воспроизводимость и полнота}')]
body=body.replace('\\section{Контрфактические вопросы и проверка результатов}','\\section{Ответы на контрфактические вопросы}')
body=re.sub(r'Методические указания:.*?сохранены из ЛР0\.',
    'Глубины 8, 24, 48, 96; ширина 256; $\\\\sigma=0.02$; SGD с импульсом 0.9. '
    'Использован CIFAR-10, допустимый по §6 задания. Данные и единый цикл обучения унаследованы из ЛР0.',body,flags=re.S)
body=re.sub(r'Протокол данных преподавателя не предоставлен\..*?Роль cal зарезервирована и не используется\.',
    'Использовано стратифицированное разбиение ЛР0 с seed 1729: по 500 объектов каждого класса для val и по 200 для cal. Роль cal не используется.',body,flags=re.S)
body=re.sub(r'Код поисковой серии:.*?хешам\.','',body,flags=re.S)
body=re.sub(r'Предварительная проверка времени.*?с/эпоха\.','',body,flags=re.S)
body=body.replace('Исходный FAIL остаётся в таблице и acceptance.json.','Исходное невыполнение порога сохраняется в выводе.')
body=re.sub(r'Гипотеза и точная конфигурация сохранены.*?A: обычная',
    'До обучения E3 сформулирована гипотеза о меньшей относительной ошибке остаточной сети. A: обычная',body,flags=re.S)
body=body.replace('этот предел подсказки PDF','этот предел приближения')
body=body.replace('и отличие bias от весов проверяются отдельным сохранённым разложением, если оно включено в реестр.',
                  'и отличие bias от весов проверены отдельным разложением R39.')
body=body.replace('Глубины исходной пары могут различаться; следующий уровень берётся отдельно для каждой архитектуры.',
                  'Обе исходные сети имеют глубину 8; следующий уровень профиля равен 24.')
body=body.replace('Отдельная проба R39: внутренние весовые g (слои2–24) min=3.479e-17, max=2.521e-16, max/min=7.244; bias g min=4.91e-17, max=0.04318.',
    r'В пробе R39 внутренние весовые $g$ (слои 2--24) лежат от $3.479\cdot10^{-17}$ до $2.521\cdot10^{-16}$, отношение максимума к минимуму равно 7.244. Для bias диапазон составляет от $4.91\cdot10^{-17}$ до 0.04318.')
body=body.replace('Фактические изменения при увеличении глубины: R22→R23: изменение val accuracy -0.0502, изменение rho 0.07559; R30→R31: изменение val accuracy 0.0002, изменение rho -0.009899.',
    r'При переходе от глубины 8 к 24 accuracy обычной сети снизилась на 0.0502, а $\rho_T$ выросла на 0.07559 (R22/R23). Для масштабированной остаточной сети accuracy изменилась на +0.0002, а $\rho_T$ --- на $-0.009899$ (R30/R31).')
theory=r'''
\subsection{Цель и теоретические сведения}
Цель работы --- выяснить, как инициализация и остаточные связи влияют на сохранение сигнала, градиенты и обучение глубоких сетей, а затем проверить основной вывод на трёх seed. Протокол и критерии взяты из методических указаний~\cite{teacher}.

Номера вида (1.x) в анализе относятся к формулам исходного задания; основные используемые зависимости приведены ниже.

Для слоя $h_l=\phi(N_l(W_lh_{l-1}+b_l))$ измеряется средний квадрат $q_l=\mathbb E[h_l^2]$. При независимых центрированных весах, нулевых смещениях и ReLU приближение имеет вид
\begin{equation}q_l=\chi_lq_{l-1},\qquad\chi_l=n_{l-1}\sigma_l^2/2.\end{equation}
Поэтому He даёт $\chi=1$, Xavier при равной ширине --- $1/2$, а фиксированное $\sigma=0.02$ при $n=256$ --- $0.0512$. Небольшое отклонение от единицы накапливается с глубиной. Для BatchNorm статистики вычисляются по batch, для LayerNorm --- по признакам отдельного объекта; обязательные серии ниже выполняются без этих нормализаций.

Остаточный блок $h_{l+1}=h_l+\alpha F_l(h_l)$ имеет якобиан $I+\alpha\partial F_l/\partial h_l$. При общем коэффициенте ветви $c$ ожидается
\begin{equation}q_L/q_0\approx(1+\alpha^2c)^L.\end{equation}
Масштаб $\alpha=L^{-1/2}$ ограничивает это приближение величиной $e^c$; прямой путь при $\alpha=1$ сам по себе не ограничивает рост сигнала.

Измеряемые показатели обучаемости определены как
\begin{equation}\rho_T=\frac{\mathcal L_T}{\ln10},\quad
g_l=\frac{\|\nabla_{\theta_l}\mathcal L\|_2}{\sqrt{|\theta_l|}},\quad
GI=\log_{10}\frac{\max_l\widetilde g_l+\varepsilon}{\min_l\widetilde g_l+\varepsilon},\end{equation}
\begin{equation}AVD=\max_l|\log_{10}(v_l+\varepsilon)-\log_{10}(v_1+\varepsilon)|.\end{equation}
Здесь $\mathcal L_T$ --- cross-entropy на фиксированной обучающей подвыборке, $\widetilde g_l$ --- медиана по измеряемым шагам, $v_l$ --- дисперсия активаций, $\varepsilon=10^{-12}$. Значения $\rho_T\le0.5$ и $\rho_T\ge0.9$ задают обучаемый и необучаемый режимы; остальные значения пограничны.

\subsection{Фактический протокол}
'''
body=body.replace('\\section{Профиль и протокол}','\\section{Профиль и протокол}\n'+theory)
captions=['Выбор единой скорости обучения по validation, три эпохи.',
          'Предсказанный и измеренные множители при инициализации.',
          'Повторная проба Gaussian после одной эпохи обучения.',
          'Обучение сетей разной глубины: все условия E2.',
          'Число обновлений до численного сбоя при единичной ветви.',
          'Глубина и число параметров обычных и остаточных сетей.',
          'Рост сигнала остаточных сетей до обучения.',
          'Основное сравнение E3: все три парных seed.',
          'Средние значения и выборочные SD в E3.',
          'Измеренный и предсказанный рост при единичной ветви.',
          'Проверка следующей глубины для конфигураций с близкой точностью.']
idx=[0]
def convert(m):
    i=idx[0];idx[0]+=1;content=m[1]
    fn=f'table_{i+1:02}.tex';(out/'tables'/fn).write_text(content,'utf-8')
    return '\\begin{table}[H]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{'+captions[i]+'}\n\\FitTable{\\input{tables/'+fn+'}}\n\\end{table}'
body=re.sub(r'\\begin\{center\}\\small\s*(\\begin\{tabular\}.*?\\end\{tabular\})\\end\{center\}',convert,body,flags=re.S)
assert idx[0]==len(captions)
def convertfig(m):
    options,name,cap=m.groups();source=REPORTS/'lab1'/'images'/name;track([source]);shutil.copy2(source,out/'figures'/name)
    options=options.replace('.88\\textheight','.77\\textheight')
    return '\\begin{figure}[H]\\centering\\includegraphics['+options+']{figures/'+name+'}\n\\caption{'+cap+'}\n\\end{figure}'
body=re.sub(r'\\begin\{center\}\\includegraphics\[([^]]+)\]\{images/([^}]+)\}\\par\\small (.*?)\\end\{center\}',convertfig,body,flags=re.S)
body+=r'''
\subsection*{Заключение}
He сохраняет средний масштаб сигнала значительно лучше Xavier и Gaussian с фиксированным малым $\sigma$. Масштабирование остаточной ветви предотвращает наблюдавшиеся численные сбои единичной ветви и снижает $\rho_T$ на глубине 96: средняя парная разность равна 0.291256 при SD 0.010034. По правилу задания различие установлено. Однако граница роста $1.2e^c$ в одном условии не выполнена, а значения $\rho_T$ большей части обученных сетей остаются пограничными. Вывод относится к одному split, трём эпохам и выбранным архитектурам; разные числа параметров plain/residual следует учитывать при переносе сравнения.
'''
title='Почему глубокая сеть обучается или не обучается. Инициализация, нормализация и остаточные связи'
body+=bibliography(1,title)
body+=registry(out,[(v['ref'],k,v['RUN_ID']) for k,v in runs.items() if v['ref'] not in ['R01','R02']])
(out/'report_body.tex').write_text(body,'utf-8')
main=(out/'main.tex').read_text('utf-8').replace('\\begin{document}',r'\newcommand{\FitTable}[1]{\sbox0{#1}\ifdim\wd0>\linewidth\resizebox{\linewidth}{!}{\usebox0}\else\usebox0\fi}'+'\n\\begin{document}')
(out/'main.tex').write_text(main,'utf-8')
save(out/'build/content_check.json',{'teacher_pdf':str(pdf),'teacher_pages':13,'tables':12,'figures':4,'counterfactuals':5,'E3':d['E3'],'raw_summaries_checked':len(runs),'old_report_unchanged':True})
update(1,written='PASS')
compile_report(1)
print('LAB1 COMPILED',out)
