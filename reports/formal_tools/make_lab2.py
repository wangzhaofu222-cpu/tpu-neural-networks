from common import *
import csv,statistics
out=setup(2,['Архитектурные inductive biases в компьютерном зрении',
 'CNN vs Vision Transformer: sample efficiency,','augmentation, transfer и distribution shift'])
p=ROOT/'experiments/lab2';d=load(p/'results.json');cf=load(p/'counterfactuals.json')
track([p/f for f in ['results.json','counterfactuals.json','E1.csv','E2.csv','E3.csv','selected_lr.json','selection.json','corruption_protocol.json']])
track([next((ROOT/'drive-download-20261026T053112Z-1-001').glob('lab2_*.pdf'))])
reg=[];ids={}
def alias(rid,label):
    if rid not in ids:
        a=f'R{len(ids)+1:02}';ids[rid]=a;reg.append((a,label,rid))
    return ids[rid]
for r in d['E1']+d['E3_rows']:
    alias(r['RUN_ID'],f"{r['model'].upper()} r{r['fraction']} s{r['seed']}")
    s=ROOT/'runs/lab2'/r['RUN_ID']/'summary.json';track([s]);q=load(s)
    assert q['global_step']==10000
e1=table(out,'e1','E1: validation и test при общем бюджете 10 000 обновлений.',
    ['Модель','$r$','$n_r$','Val acc.','Test acc.','$E_r$','Источник'],
    [[r['model'].upper(),r['fraction'],r['n_train'],f"{r['validation_accuracy']:.4f}",f"{r['test_accuracy']:.4f}",f"{r['data_epochs']:.2f}",ids[r['RUN_ID']]] for r in d['E1']], 'lrrrrrl')
resources=cf['CF3']['resources']
resource=table(out,'resources','Ресурсы моделей: FP32, CUDA; время обучения при $r=1$.',
 ['Модель','Параметры','MAC, млн','Latency, мс','Объект/с','Train MiB','Train, с'],
 [[m.upper(),r['parameters'],f"{r['MACs']/1e6:.2f}",f"{r['latency_B1_seconds']*1000:.3f}",
   f"{r['throughput_B128_per_second']:.1f}",f"{r['training_peak_gpu_memory_bytes']/2**20:.1f}",f"{r['training_seconds']:.2f}"] for m,r in resources.items()], 'lrrrrrr', 'footnotesize')
rows=list(csv.DictReader((p/'E2.csv').open(encoding='utf-8-sig')))
e2rows=[]
for c in ['blur','contrast','patch_shuffle']:
 for s in range(4):
    a=next(r for r in rows if r['corruption']==c and int(r['level'])==s and r['model']=='cnn')
    b=next(r for r in rows if r['corruption']==c and int(r['level'])==s and r['model']=='vit')
    e2rows.append([c.replace('_',' '),s,f"{float(a['accuracy']):.4f}",f"{float(b['accuracy']):.4f}",
                  f"{float(a['relative_drop']):.4f}",f"{float(b['relative_drop']):.4f}"])
e2=table(out,'e2','E2: точность и относительное падение качества на test; веса полных моделей фиксированы.',
 ['Искажение','$s$','$M_{CNN}$','$M_{ViT}$','$\delta_{CNN}$','$\delta_{ViT}$'],e2rows,'lrrrrr')
means=table(out,'means','Среднее качество по трём ненулевым уровням и смена лидера.',
 ['Искажение','$\overline M_{CNN}$','$\overline M_{ViT}$','$\Delta$','Смена'],
 [[r['corruption'].replace('_',' '),f"{r['cnn_mean']:.6f}",f"{r['vit_mean']:.6f}",f"{r['delta']:+.6f}",'да' if r['leader_reversal'] else 'нет'] for r in d['E2']],'lrrrl')
e3rows=[]
for s in [1,2,3]:
 a=next(r for r in d['E3_rows'] if r['seed']==s and r['model']=='cnn');b=next(r for r in d['E3_rows'] if r['seed']==s and r['model']=='vit')
 e3rows.append([s,f"{a['test_accuracy']:.4f}",f"{b['test_accuracy']:.4f}",f"{b['test_accuracy']-a['test_accuracy']:+.6f}",ids[a['RUN_ID']]+'/'+ids[b['RUN_ID']]])
e3=table(out,'e3','E3: все парные результаты на доле $r=0.05$.',['Seed','CNN','ViT','$\Delta_s$','Источники'],e3rows,'lrrrl')
assert abs(statistics.mean(d['E3']['deltas'])-d['E3']['mean'])<1e-12
assert abs(statistics.stdev(d['E3']['deltas'])-d['E3']['sample_std'])<1e-12
trans=table(out,'translation','Дополнительная оценка сдвигов без переобучения полных моделей.',
 ['Сдвиг, пикс.','CNN accuracy','ViT accuracy','Падение CNN','Падение ViT'],
 [[s,*[f"{next(r for r in cf['CF5']['translation_results'] if r['pixels']==s and r['model']==m)['accuracy']:.4f}" for m in ['cnn','vit']],
   *[f"{next(r for r in cf['CF5']['translation_results'] if r['pixels']==s and r['model']==m)['absolute_drop']:+.4f}" for m in ['cnn','vit']]] for s in [2,4,8]],'rrrrr')
figs={name:figure(out,p/'figures'/fn,name,cap,h) for name,fn,cap,h in [
 ('learning','E1_learning_curves.png','Test accuracy и разность ViT минус CNN в зависимости от доли данных; источники приведены в таблице E1.','.38'),
 ('loss','training_loss.png','Сохранённые train loss всех условий E1 при одинаковом числе обновлений.','.47'),
 ('time','time_accuracy.png','Время обучения и test accuracy при полном train; два исходных запуска E1.','.30'),
 ('corruptions','E2_corruptions.png','Изменение точности при трёх искажениях персонального профиля; веса не переобучаются.','.39'),
 ('paired','E3_paired_seeds.png','Все три парные разности E3. Положительное значение означало бы преимущество ViT.','.28'),
 ('map','trainability_map.png','Карта относительного преимущества: доля train и условие test; фиксированы модели, A1, бюджет и seed 1.','.47')
]}
body=r'''\section{Профиль и протокол}
\subsection{Цель и теоретические сведения}
Цель работы --- сравнить индуктивные смещения CNN и ViT при сопоставимом числе параметров, изменении объёма train и искажении test. Методика соответствует заданию~\cite{teacher}. Локальное ядро и общие веса CNN задают предположение о пространственной структуре изображения; ViT получает глобальные связи через внимание и обучаемые позиционные вложения.

Для свёртки с ядром $k$, каналами $C,C'$ и выходом $H'\times W'$ число MAC равно $H'W'C'Ck^2$. ViT преобразует патчи $P\times P$ в токены ширины $D$, добавляет CLS и позиционные вложения. Длина последовательности $N'=HW/P^2+1$; в каждом из шести pre-LN блоков применяются внимание и MLP с GELU:
\begin{equation}\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt{d_k})V,\quad
C_{block}\approx12N'D^2+2N'^2D.\end{equation}
Здесь $Q,K,V$ --- проекции токенов, $d_k$ --- ширина головы. Квадратичный член объясняет чувствительность стоимости к размеру патча.

Основная метрика --- accuracy; относительное преимущество и падение при искажении заданы как
\begin{equation}\Delta(r)=M_{ViT}(r)-M_{CNN}(r),\quad
\delta_c(s)=\frac{M_{clean}-M_c(s)}{M_{clean}+10^{-12}},\quad
\overline M_c=\frac13\sum_{s=1}^3M_c(s).\end{equation}
Искажение $c$ и уровень $s$ не меняют метки. Для смены лидера обе разности, на чистом test и по $\overline M_c$, должны иметь противоположные знаки и модуль не менее 0.01.

\subsection{Данные и условия обучения}
Использован CIFAR-10, разбиение ЛР0: train 43 000, val 5 000, cal 2 000, test 10 000. Нормализация рассчитана только по train; cal не используется. Вложенные стратифицированные подмножества заданы профилем P3adf02: $r=0.05,0.15,0.30,0.60,1$. Индексы каждого класса один раз перемешаны генератором базового seed. Изменение $r$ только добавляет объекты.

CNN --- ResNet-18 со свёрткой stem $3\times3$, stride 1 и без начального max-pool. ViT имеет $P=4$, $D=384$, шесть блоков и шесть голов, MLP ширины 1536, CLS и обучаемые абсолютные позиции. Предобучение не используется. Общие условия: AdamW, weight decay 0.05, batch 128, ровно $S=10\,000$ обновлений, 5\% прогрева и cosine, FP32 на RTX 3060 Laptop. LR выбрана отдельно по val на полном train из $0.0003,0.001,0.003$: CNN 0.003, ViT 0.0003. Сравнивается архитектура вместе с этим заранее фиксированным рецептом обучения.

Политика A1 --- случайный crop $32\times32$ после padding 4 и горизонтальное отражение. Профиль содержит A2 RandAugment+CutMix, но дополнительные обучающие серии A0/A2 здесь не заявляются. Персональные искажения E2: blur, contrast, patch shuffle.

\section{Ресурсы моделей}
Режим сравнения --- param matched. Число параметров ViT составляет 95.72\% числа параметров CNN, что укладывается в допуск 20\%. Число MAC ViT больше в 1.27965 раза, поэтому равенство вычислений не установлено. Подсчёт включает QKV, выходную проекцию и произведения attention; аналитическая и измеренная суммы совпали.
@@RES@@
Latency при $B=1$ и throughput при $B=128$ получены из 30 повторов после 10 прогревов с синхронизацией CUDA. Throughput в этой таблице вычислен по медианному времени собственного batch. Пиковая память относится к обучению. Малый batch CNN показал повышенный разброс времени, поэтому его медиана характеризует данную серию, а не универсальную задержку. При полном train ViT обучался в 1.54076 раза дольше CNN.
@@TIME@@
\section{E1. Эффективность по объёму данных}
Теоретическое ожидание --- преимущество локальности CNN при малом числе примеров и возможное сокращение разрыва при росте train. Для каждой доли обе модели выполняют ровно 10 000 обновлений, поэтому число проходов по малой выборке больше:
\begin{equation}E_r\approx S/\lceil n_r/B\rceil.\end{equation}
$n_r$ --- число обучающих изображений, $B=128$. В таблице показано фактически обработанное число эквивалентных эпох; различия последнего неполного batch учитываются журналом.
@@E1@@
@@LEARNING@@
Разность ViT минус CNN изменилась от $-0.2572$ при $r=0.05$ до $-0.1259$ при $r=1$. Разрыв сократился на 0.1313, но лидер не сменился ни на одной доле; $r^*$ в измеренном диапазоне отсутствует. Нельзя продолжать кривую за $r=1$ без новых данных. Увеличение данных здесь одновременно уменьшает число повторных проходов при фиксированном $S$.
@@LOSS@@
\section{E2. Искажения тестовых изображений}
Использованы фиксированные полные модели E1, CNN R09 и ViT R10. Все преобразования выполнены в RGB $[0,1]$ до нормализации, затем значения ограничены этим диапазоном. Blur: $\sigma=0.5,1,1.5$ и ядра 5,7,11; contrast: коэффициенты 0.6,0.4,0.2 относительно средней яркости; patch shuffle: стороны блока 16,8,4. При $s=0$ вход тождествен чистому. Обе архитектуры получают одни и те же случайные преобразования каждого test-изображения.
@@E2@@
@@MEANS@@
@@CORRUPTIONS@@
На чистом test преимущество CNN равно 0.1259. Для blur средняя разность ViT минус CNN положительна, 0.049067, для patch shuffle --- 0.176033: в обоих случаях выполнено правило смены лидера. Contrast сохраняет преимущество CNN, средняя разность $-0.314733$. Это показывает зависимость качества от типа сдвига. Перестановка блоков разрушает локальную структуру CNN; для ViT она меняет позиции токенов, но не содержимое патча $4\times4$. Обучаемые абсолютные позиции не позволяют объявить ViT строго инвариантным к перестановке.

\section{E3 и карта относительного преимущества}
Режим E3 выбран по максимальному модулю разности на val: $r=0.05$, исходная разность $-0.2568$. До дополнительных seed зафиксирована гипотеза о сохранении отрицательного знака и близкой величины на test. Одинаковы подмножество, A1, $S$, batch, архитектуры и выбранные LR; меняются только общие seed 1,2,3. Seed 1 соответствует E1 и явно повторно используется.
@@E3@@
Для $\Delta_s=M_{ViT,s}-M_{CNN,s}$ среднее равно $-0.249467$, выборочное SD --- 0.017253. Все три разности отрицательны, а $|\overline\Delta|>2s_\Delta=0.034506$: различие установлено в пользу CNN. Отклонение среднего от численной гипотезы $-0.2568$ составляет +0.007333. Направление ожидания подтверждено.
@@PAIRED@@
@@MAP@@
В карте меняются доля train и условие test; фиксированы A1, бюджет, рецепты оптимизации и seed 1. Это двумерный срез, а не сравнение после подбора по test. Три seed E3 описывают вариативность обучения на одном подмножестве; смену split и рецепта они не охватывают.

\section{Ответы на контрфактические вопросы и выводы}
\subsection{1. Рост объёма данных}
Разность изменилась с $-0.2572$ до $-0.1259$ (R01/R02 и R09/R10), то есть на +0.1313. Это сокращение разрыва не предсказывает знак при большем датасете: train CIFAR-10 исчерпан. Локальность, общие веса, A1, конечный бюджет и выбранные LR могут сохранять преимущество CNN; эксперимент не выделяет один из этих механизмов отдельно.
\subsection{2. Одинаковое число эпох вместо одинаковых шагов}
На полном train $E=10\,000/336=29.7619$. Для 2150 изображений это даёт $\lceil29.7619\cdot17\rceil=506$ шагов. Сохранённый дополнительный опыт CNN дал test accuracy 0.6169 против 0.7074 при 10 000 шагах: падение 0.0905. Меняется и число предъявлений, и горизонт cosine с прогревом, поэтому это не чистая оценка одного механизма. Источник: @@CF2ID@@.
\subsection{3. Параметры и вычислительный бюджет}
По исходным профилям отношение параметров ViT/CNN равно 0.957186, MAC --- 1.279649, времени обучения --- 1.540765. Режим равных параметров выполнен, равных вычислений --- нет. Число параметров не определяет цену внимания и пропускную способность.
\subsection{4. Уменьшение патча вдвое}
Для $P=4$ имеется $N'=65$, для $P=2$ --- 257 токенов. Стоимость блока выросла с 118 260 480 до 505 479 936 MAC; квадратичный член --- в $(257/65)^2=15.632899$ раза. Полная модель с $P=2$ имеет 10 755 466 параметров, лишь в 1.005601 раза больше исходной. Размеры весов блоков зависят от $D$, а не числа токенов; меняются patch projection и таблица позиций. Значения подтверждены сохранённым профилированием @@CF4ID@@, обучения этой модификации не заявляется.
\subsection{5. Сдвиги на 2, 4 и 8 пикселей}
@@TRANS@@
Сдвиги 4 и 8 кратны патчу ViT, только 8 кратен общему stride CNN. Однако 8 пикселей дают наибольшее среднее падение, 0.0523, против $-0.0008$ и 0.0080 при 2 и 4. A1 покрывает сдвиги 2/4, но не 8; границы изображения, padding и pooling также нарушают строгую инвариантность классификации. Поэтому простая кратность stride не объясняет результат, а вклад аугментации нельзя отделить от величины сдвига по этим трём точкам.
\noindent\textbf{Заключение.} CNN точнее ViT на всех долях train, а при 5\% данных это подтверждено на трёх seed. Blur и patch shuffle меняют лидера. Преимущество зависит от условий; равенство параметров не означает равенства вычислений.
'''
replacement={'RES':resource,'E1':e1,'E2':e2,'MEANS':means,'E3':e3,'TRANS':trans,
             'CF2ID':alias(cf['CF2']['RUN_ID'],'CF2 CNN 506 steps'),
             'CF4ID':alias(cf['CF4']['P2_RUN_ID'],'CF4 P2 profiling')}
replacement.update({k.upper():v for k,v in figs.items()})
for k,v in replacement.items():body=body.replace('@@'+k+'@@',v)
assert '@@' not in body
body=re.sub(r'\\subsection\{([1-5])\. ',r'\\subsection*{Вопрос \1. ',body)
body+=bibliography(2,'Архитектурные inductive biases в компьютерном зрении. CNN vs Vision Transformer: sample efficiency, augmentation, transfer и distribution shift')
body+=registry(out,reg)
(out/'report_body.tex').write_text(body,'utf-8')
save(out/'build/content_check.json',{'teacher_pages':14,'counterfactuals':5,'training_executed':False,'E3':d['E3'],'source_files_preserved':True})
compile_report(2)
print('LAB2 COMPILED')
