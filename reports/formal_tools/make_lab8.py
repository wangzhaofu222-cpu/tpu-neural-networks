from common import *
import csv
out=setup(8,['Когда предсказанию сети можно доверять','Надёжность, калибровка и неопределённость'])
p=ROOT/'experiments/lab8';cf=load(p/'counterfactual_answers.json');sources=load(p/'model_sources.json')
track([p/f for f in ['counterfactual_answers.json','E1.csv','E1_binning.csv','E2_grid.csv','E2_pairs.csv','E2_T5.csv','E3.csv','selective_ood.csv','model_sources.json','hypothesis.md','profile.json']])
track([next((ROOT/'drive-download-20261026T053112Z-1-001').glob('lab8_*.pdf'))])
reg=[];ids={}
def alias(rid,label):
 if rid not in ids:ids[rid]=f'R{len(ids)+1:02}';reg.append((ids[rid],label,rid))
 return ids[rid]
def rows(f):return list(csv.DictReader((p/f).open(encoding='utf-8-sig')))
source=table(out,'sources','Использованные источники моделей ЛР2: доля train 0.05, 10 000 обновлений.',
 ['Модель','Seed','Архитектура','Источник'],[[k.split('_')[0].upper(),k.split('_')[1],'ResNet-18' if k.startswith('cnn') else 'ViT P4 D384',alias(sources[k]['RUN_ID'],'Lab2 '+k)] for k in ['cnn_1','cnn_2','cnn_3','vit_1']],'ll ll')
e1=table(out,'e1','E1: чистый test; ECE --- 15 интервалов равной ширины.',
 ['Модель','$T$','Accuracy','Conf.','ECE','NLL','Brier','Пусто','Ист.'],
 [[r['model'].upper(),f"{float(r['temperature']):.5f}",*[f"{float(r[k]):.5f}" for k in ['accuracy','mean_confidence','ECE','nll','brier']],r['ECE_empty_bins'],alias(r['RUN_ID'],'E1 '+r['model'])] for r in rows('E1.csv')],'lrrrrrrrl','scriptsize')
binning=table(out,'binning','Граница E1: другие схемы ECE, те же предсказания.',
 ['Модель','M','Интервалы','$T$','ECE','Пусто'],
 [[r['model'].upper(),r['M'],'равной ширины' if r['binning']=='equal_width' else 'равное число',f"{float(r['temperature']):.4f}",f"{float(r['value']):.6f}",r['empty_bins']] for r in rows('E1_binning.csv')],'ll lrrr','footnotesize')
gr=[r for r in rows('E2_grid.csv') if r['source_key']=='cnn_1'];grows=[]
names={'gaussian_noise':'Шум','gaussian_blur':'Размытие','brightness':'Яркость','contrast':'Контраст','occlusion':'Закрытие'}
for c in names:
 for l in range(6):
  a=next(r for r in gr if r['corruption']==c and int(r['level'])==l and float(r['temperature'])==1)
  b=next(r for r in gr if r['corruption']==c and int(r['level'])==l and float(r['temperature'])!=1)
  grows.append([names[c],l,f"{float(a['accuracy']):.4f}",f"{float(a['mean_confidence']):.4f}",f"{float(b['mean_confidence']):.4f}",f"{float(a['ECE']):.4f}",f"{float(b['ECE']):.4f}",f"{float(a['nll']):.4f}",f"{float(b['nll']):.4f}"])
grid=table(out,'grid','E2: CNN seed 1, общий test; индексы 1 и T обозначают температуры 1 и $T^*$.',
 ['Искажение',r'$\ell$','Acc.',r'$Conf_1$',r'$Conf_T$',r'$ECE_1$',r'$ECE_T$',r'$NLL_1$',r'$NLL_T$'],grows,'lr'+'r'*7,'scriptsize')
pairrows=rows('E2_pairs.csv')
pairs=table(out,'pairs','Основное сравнение E2: выигрыш температуры на clean минус occlusion5.',
 ['Seed',r'$T^*$',r'$g_A$',r'$g_B$',r'$\Delta_s$','Ист. clean/shift'],
 [[r['seed'],*[f"{float(r[k]):.6f}" for k in ['temperature','gA','gB','delta']],alias(r['clean_RUN_ID'],'E2 clean s'+r['seed'])+'/'+alias(r['shift_RUN_ID'],'E2 occ5 s'+r['seed'])] for r in pairrows],'rrrrrl','footnotesize')
t5=table(out,'t5','Температура, подобранная на corrupted val, и её перенос на два test-условия.',
 ['Test','$T$','Accuracy','Conf.','ECE','NLL','Brier','Пусто'],
 [[r['condition'],f"{float(r['temperature']):.5f}",*[f"{float(r[k]):.5f}" for k in ['accuracy','mean_confidence','ECE','nll','brier']],r['ECE_empty_bins']] for r in rows('E2_T5.csv')],'lrrrrrrr','footnotesize')
attacks=[r for r in rows('E3.csv') if float(r['temperature'])==1]
attack=table(out,'attacks','E3: T=1, один и тот же полный test, 10 000 изображений.',
 ['Модель','Атака',r'$255\epsilon$','K/R','Acc.','Conf.','ECE','Ист.'],
 [[r['source_key'].replace('_',' '),r['attack'],r['eps_255'],r['K']+'/'+r['R'],*[f"{float(r[k]):.5f}" for k in ['accuracy','mean_confidence','ECE']],alias(r['RUN_ID'],r['source_key']+' '+r['attack']+' eps'+r['eps_255']+' K'+r['K'])] for r in attacks],'lll lrrrl','footnotesize')
oodrows=rows('selective_ood.csv')
ood=table(out,'ood','Selective и semantic OOD: меньше AURC и FPR95, больше AUROC/AUPR лучше.',
 ['Модель','$T$','Мера','AURC','Избыток','AUROC','AUPR','FPR95'],
 [[r['source_key'].replace('_',' '),f"{float(r['temperature']):.3f}",r['measure'],*[f"{float(r[k]):.4f}" for k in ['AURC','excess_AURC','AUROC','AUPR','FPR95']]] for r in oodrows],'ll lrrrrr','scriptsize')
for r in oodrows:alias(r['RUN_ID'],'selective OOD '+r['source_key'])
figspec=[('cnn','E1_cnn.png','CNN: reliability diagram и confidence до/после T*=2.956378; ECE 15 equal-width.','.37'),
 ('vit','E1_vit.png','ViT: аналогичная оценка, T*=4.273313, ECE 15 equal-width.','.37'),
 ('bins','E1_binning.png','ECE при изменении схемы интервалов для обеих моделей.','.32')]
for c in names:figspec.append((c,'E2_'+c+'.png',f'E2: {names[c]}, CNN seed 1; ECE 15 equal-width, T=1 и T*=2.956378.','.34'))
figspec += [('paired','E2_pairs.png','Три парные разности E2; положительный знак ожидался до измерения.','.30'),
 ('attackfig','E3_attacks.png','FGSM eps=0,1,2,4,8 /255 и PGD eps=2/255, K10 R1 и K20 R2, alpha=2.5 eps/K.','.37'),
 ('examplecnn','E3_examples_cnn.png','CNN: исходные и атакованные изображения, усиленное возмущение и его Linf-норма.','.42'),
 ('examplevit','E3_examples_vit.png','ViT: соответствующие примеры на тех же test ID.','.42'),
 ('riskcnn','selective_ood_cnn_1.png','CNN seed 1: risk--coverage и OOD при T=1 / T*=2.956378.','.37'),
 ('riskvit','selective_ood_vit_1.png','ViT seed 1: risk--coverage и OOD при T=1 / T*=4.273313.','.37'),
 ('oodcnn','OOD_measures_cnn_1.png','Распределения трёх мер неопределённости CNN на CIFAR-10 и SVHN.','.36'),
 ('oodvit','OOD_measures_vit_1.png','Распределения трёх мер неопределённости ViT на CIFAR-10 и SVHN.','.36')]
figs={n:figure(out,p/'figures'/f,n,c,h) for n,f,c,h in figspec}
body=r'''\section{Профиль и протокол}
\subsection{Цель и определения}
Цель --- оценить, когда уверенность замороженной модели соответствует ошибкам и где она перестаёт быть надёжной~\cite{teacher}. Accuracy и калибровка --- разные свойства. Для вероятностей $p=\operatorname{softmax}(z)$ уверенность $conf=\max_cp_c$, а калибровка требует $P(\hat Y=Y\mid conf=p)=p$. Использованы
\begin{equation}\begin{aligned}ECE=\sum_m\frac{|B_m|}{N}|acc(B_m)-conf(B_m)|,\quad NLL=-\frac1N\sum_i\log p_{y_i},\\ BS=\frac1N\sum_{i,c}(p_{ic}-1[y_i=c])^2.\end{aligned}\end{equation}
ECE зависит от интервалов и учитывает верхнюю вероятность; NLL и Brier используют весь вектор. Пустые интервалы не дают определённой accuracy и исключаются из суммы.

Температура $p^{(T)}=\operatorname{softmax}(z/T)$ при $T>0$ сохраняет argmax, но меняет вероятности. $\ln T$ минимизирует NLL только на val, с кандидатом T=1. Основные меры неопределённости, больше означает менее уверенно:
\begin{equation}U_{MSP}=1-\max_cp_c,\qquad H=-\sum_cp_c\ln p_c,\qquad U_{margin}=1-p_{(1)}+p_{(2)}.\end{equation}

\subsection{Данные, модели и искажения}
Профиль: CNN, основное искажение occlusion, PGD $\epsilon=2/255$. ECE: 15 интервалов равной ширины. Использованы сохранённые модели ЛР2, обученные на 2150 train-изображениях (доля 0.05) за 10 000 обновлений. Для CNN имеются три подлинных seed, поэтому test не делится на псевдонезависимые seed. ViT seed 1 служит вторым архитектурным ориентиром. Новые классификаторы не обучаются.
@@SOURCE@@
CIFAR-10: train 43 000, val 5000, cal 2000, test 10 000. Нормализация взята из train ЛР2 и применяется ровно один раз. T подбирается на val; cal не используется, поскольку дополнительное conformal prediction не выполняется. Все атаки и оценки используют общий полный test. Устройство --- RTX 3060 Laptop. Для semantic OOD использован полный test SVHN, 26 032 изображения, без смысловых меток цифр.

Реализованы пять локальных параметризованных искажений уровней 0--5. Gaussian noise: std 0,0.04,0.06,0.08,0.09,0.10; Gaussian blur: sigma 0,0.4,0.7,1.0,1.3,1.6; brightness: множители 1,1.15,1.30,1.45,1.60,1.75; contrast: 1,0.8,0.6,0.4,0.2,0.1; occlusion: доля площади 0,0.1,0.2,0.3,0.4,0.5. Это явные локальные приближения, а не сертифицированная реализация CIFAR-10-C. Они работают до нормализации в RGB [0,1], сохраняют форму и метку; уровень 0 тождественный.

\section{E1. Калибровка и температура}
@@E1@@
Все 10 000 предсказанных классов каждой модели сохранены: разность accuracy равна нулю. CNN val NLL уменьшилась с 1.602649 до 0.890119. На test обе модели улучшили ECE, NLL и Brier. До масштабирования CNN одновременно точнее и лучше откалибрована; после него порядок сохраняется. Это наблюдаемая связь двух моделей, а не тождество точности и калибровки.
@@CNN@@
@@VIT@@
@@BINNING@@
@@BINS@@
Вывод о снижении ECE сохраняется при 5 и 50 интервалах равной ширины и при 15 интервалах с равным числом объектов. Значения при разных схемах сравниваются только внутри одной схемы. Они не доказывают идеальную калибровку каждого отдельного класса или объекта.

\section{E2. Перенос температуры на искажённые данные}
CNN seed 1 исследована при всех пяти типах и шести уровнях; веса и T* с чистой val фиксированы. В таблице значения после масштабирования помечены индексом T. Accuracy одинакова при обеих температурах.
@@GRID@@
@@GAUSSIAN_NOISE@@
@@GAUSSIAN_BLUR@@
@@BRIGHTNESS@@
@@CONTRAST@@
@@OCCLUSION@@
При occlusion5 accuracy падает с 0.7074 до 0.2447, тогда как исходная confidence --- с 0.922246 до 0.752771. Модель становится гораздо менее правильной, оставаясь уверенной. Температура смягчает overconfidence, но не восстанавливает классы.

До основного сравнения ожидалось $\Delta_s=g_A(s)-g_B(s)>0$, где $g=ECE_1-ECE_{T^*}$, A --- clean, B --- occlusion5. Единственный фактор пары --- входное условие; для каждого seed одна модель и T*, подобранная на его чистой val. Три исходных checkpoint задают вариативность обучения, а не случайное разбиение test.
@@PAIRS@@
@@PAIRED@@
Все разности отрицательны, среднее $-0.162050322$, выборочное SD 0.010897215; модуль среднего превышает $2s_\Delta=0.021794431$. Установлен обратный вывод: в этих условиях абсолютный выигрыш температуры больше на сильном искажении. Предположение о порядке выигрышей опровергнуто. Причина согласуется с намного большей исходной избыточной уверенностью на искажении. При этом итоговая ECE после T* на occlusion5 0.168598 хуже clean 0.028418, так что больший выигрыш не означает лучшую конечную калибровку.

На исследованной сетке ни одного $g\le0$ не найдено. Граница отсутствия пользы температуры лежит вне измеренных условий, если она существует. Для известного сдвига отдельно подобрана $T_5=6.108734$ на occlusion5 val.
@@T5@@

\section{E3. Состязательные возмущения}
Атаки максимизируют CE исходной модели T=1 в eval. FGSM делает $x_{adv}=\Pi_{[0,1]}[x+\epsilon\operatorname{sign}\nabla_xL]$. PGD начинает в случайной точке допустимого шара, делает K шагов $\alpha=2.5\epsilon/K$ и после каждого проецирует на пересечение $[x-\epsilon,x+\epsilon]$ и [0,1]. При R стартах выбирается худший вариант для каждого изображения. Все ограничения заданы в исходных пикселях, параметры модели фиксированы.
@@ATTACK@@
@@ATTACKFIG@@
PGD10 даёт accuracy 0.2458/0.0760 для CNN/ViT против FGSM 0.2988/0.1031 при том же $\epsilon=2/255$. Усиление до K20,R2 снижает их до 0.2431/0.0754: десять шагов ещё не дали окончательной оценки. Во всех реальных примерах и шагах PGD выполнены Linf и диапазон; при нулевом epsilon результат совпадает с чистым test.
@@EXAMPLECNN@@
@@EXAMPLEVIT@@
Точность под конкретной атакой --- верхняя оценка устойчивости к наихудшему допустимому возмущению. Она не является сертификатом против любых атак. Высокая confidence после атаки может сопровождать почти всегда неверный класс.

\section{Профиль надёжности: отказ и OOD}
Модель отвечает при $U\le\tau$. Coverage --- доля принятых, risk --- доля ошибок среди них. При нулевом coverage риск не определён. AURC --- средний накопленный риск при сортировке U по возрастанию. Вместе с AURC показан избыток над oracle, ставящим ошибки в конец; это учитывает различную исходную accuracy моделей.

В OOD-задаче положительный класс --- SVHN. AUROC оценивает вероятность большего U у OOD с половинным весом ties; FPR95 --- доля ID, отклонённых при обнаружении не менее 95\% OOD. AUPR зависит от доли OOD: здесь $26032/(26032+10000)=0.72247$. Смысловые метки SVHN не используются.
@@OOD@@
@@RISKCNN@@
@@RISKVIT@@
@@OODCNN@@
@@OODVIT@@
Для CNN seed 1 MSP-AURC 0.115934 при oracle 0.047742, после T* 0.114975; AUROC 0.674099 уменьшается до 0.667306. Для ViT AURC 0.393064 уменьшается до 0.385041, AUROC 0.578933 растёт до 0.592006. Хорошая калибровка на clean не обеспечила сильного разделения semantic OOD.

На SVHN accuracy, ECE, NLL и Brier не определены: в десяти классах CIFAR-10 нет правильного класса цифры. В профиле также не заявлены conformal coverage, near-OOD и ансамбль. Оценки относятся к измеренным искажениям, epsilon и далёкому OOD; будущий неизвестный сдвиг не покрыт.

\section{Ответы на контрфактические вопросы}
\subsection*{Вопрос 1. ECE против NLL и Brier}
У CNN изменения после T*: ECE $-0.186429$, NLL $-0.730550$, Brier $-0.084769$; у ViT $-0.400536,-2.390522,-0.278635$. Все улучшились, измеренного контрпримера нет. Теоретически снижение ECE не гарантирует снижения NLL/Brier, поскольку они учитывают разные аспекты вероятностей. Val NLL не растёт из-за целевой минимизации с включённым T=1; на test такой гарантии нет.
\subsection*{Вопрос 2. T5 на clean и shift}
Относительно T* температура T5 уменьшает occlusion5 ECE с 0.168598 до 0.029181, NLL с 2.302492 до 2.143172, Brier с 0.900775 до 0.857212. На clean ECE растёт с 0.028418 до 0.265515, NLL с 0.909765 до 1.162255, Brier с 0.408943 до 0.496966. T5 соответствует известному сдвигу, но не является универсальным калибратором. Для будущего неизвестного сдвига такой val заранее нет.
\subsection*{Вопрос 3. Случайный знак при той же норме}
При $\epsilon=2/255$ случайный знак даёт accuracy CNN 0.7057 и ViT 0.4489, FGSM 0.2988/0.1031, PGD10 0.2458/0.0760. Предсказанный порядок random $>$ FGSM $\ge$ PGD подтвердился на общих ID. Случайное направление не согласовано с градиентом CE, тогда как FGSM максимизирует линейное приближение; одна норма не определяет вред возмущения.
\subsection*{Вопрос 4. Температура и уверенные ошибки}
На FGSM $2/255$ CNN confidence меняется 0.924621 $\to$ 0.717081, ECE 0.625821 $\to$ 0.418460; ViT 0.961179 $\to$ 0.572030, ECE 0.858079 $\to$ 0.468930. Изменённых предсказаний и исправленных ошибочных классов нет. Температура корректирует вероятности, а не классификационное решение.
\subsection*{Вопрос 5. Ранжирование объектов}
MSP-AURC и AUROC до/после приведены в профиле. Позиции в сортировке ID изменились у 9973 объектов CNN и 9981 ViT, классы --- у нуля. Температура сохраняет порядок логитов внутри изображения, но не обязана сохранять порядок максимальных вероятностей между изображениями. Инвариантность AUROC относится к строго возрастающему преобразованию уже готового скалярного U, а softmax(z/T) не является одним таким преобразованием MSP для всех векторов z.

\noindent\textbf{Заключение.} Температура улучшила калибровку, но не исправила классы и не гарантировала надёжность при сдвиге.
'''
repl={'SOURCE':source,'E1':e1,'BINNING':binning,'GRID':grid,'PAIRS':pairs,'T5':t5,'ATTACK':attack,'OOD':ood}
repl.update({k.upper():v for k,v in figs.items()})
for k,v in repl.items():body=body.replace('@@'+k+'@@',v)
assert '@@' not in body
body+=bibliography(8,'Когда предсказанию сети можно доверять. Надёжность, калибровка и неопределённость')+registry(out,reg)
(out/'report_body.tex').write_text(body,'utf-8');compile_report(8)
