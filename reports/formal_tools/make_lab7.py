from common import *
import csv
out=setup(7,['Diffusion и Score-Based Generative Models',
 'Noise schedule, оценка score, численный reverse process','и computational trade-offs'])
p=ROOT/'experiments/lab7';d1=load(p/'E1.json');d2=load(p/'E2.json');d3=load(p/'E3.json');diag=load(p/'diagnostics.json');cf=load(p/'counterfactual_answers.json');metric=load(p/'metrics/0.02.json')
track([p/f for f in ['E1.json','E2.json','E3.json','diagnostics.json','counterfactual_answers.json','data_protocol.json','hypothesis.md','E1_models.csv','E2_sampling.csv','E3_models.csv','supplement_rare.csv','metrics/0.02.json']])
track([next((ROOT/'drive-download-20261026T053112Z-1-001').glob('lab7_*.pdf'))])
reg=[];ids={}
def alias(rid,label):
 if rid not in ids:ids[rid]=f'R{len(ids)+1:02}';reg.append((ids[rid],label,rid))
 return ids[rid]
def rows(f):return list(csv.DictReader((p/f).open(encoding='utf-8-sig')))
e1=table(out,'e1','E1: единственное различие обучения --- расписание шума; DDPM, 5000 точек.',
 ['Расписание',r'$\bar\alpha_T$','KL','Train loss','Val loss',r'MMD$^2$',r'$D_{mode}$','Ист.'],
 [[r['schedule'],f"{float(r['alpha_bar_T']):.3e}",f"{float(r['train_mean_terminal_KL']):.3e}",f"{float(r['train_Lsimple']):.5f}",f"{float(r['val_Lsimple']):.5f}",f"{float(r['MMD2']):.7f}",f"{float(r['Dmode']):.5f}",alias(r['RUN_ID'],'E1 '+r['schedule'])] for r in rows('E1_models.csv')],'lrrrrrrl','scriptsize')
moments=table(out,'moments','Проверка q sample: максимальная по координатам ошибка среднего и относительная ошибка дисперсии.',
 ['Schedule','$t$',r'$\max|\hat m-m|$','Порог',r'$\max|\hat v/v-1|$','Порог'],
 [[k,r['t'],f"{max(r['mean_error']):.6f}",f"{r['mean_threshold']:.6f}",f"{max(r['variance_relative_error']):.6f}",f"{r['variance_relative_threshold']:.6f}"] for k,q in d1['theory'].items() for r in q['forward_moments']],'llrrrr','footnotesize')
e2rows=[r for r in rows('E2_sampling.csv') if int(r['K'])>=5 and int(r['timing_repeats'])==5]
e2=table(out,'sampling','E2: общий начальный шум; время --- медиана пяти повторов после прогрева.',
 ['Предиктор','Сэмплер','NFE',r'MMD$^2$',r'$D_{mode}$','Cov','Время, с','Ист.'],
 [[r['predictor'],r['sampler'],r['NFE'],f"{float(r['MMD2']):.7f}",f"{float(r['Dmode']):.5f}",f"{float(r['Cov']):.3f}",f"{float(r['seconds_median']):.5f}",alias(r['RUN_ID'],f"E2 {r['predictor']} {r['sampler']} K{r['K']}")] for r in e2rows],'llrrrrrl','footnotesize')
low=table(out,'low','Крайне малое число шагов DDIM; отдельная оценка, без заявления о пяти повторах времени.',
 ['Предиктор','K',r'MMD$^2$',r'$D_{mode}$','Cov','Ист.'],
 [[r['predictor'],r['K'],f"{r['MMD2']:.6f}",f"{r['Dmode']:.5f}",f"{r['Cov']:.3f}",alias(r['RUN_ID'],f"lowK {r['predictor']} K{r['K']}")] for r in d2['low_K']],'llrrrl')
fields=table(out,'fields',r'Score на сетке $41\times41$: средние с весами $p_t(x)$.',
 ['Schedule','$t$','Косинус','Отношение длин','Ист.'],
 [[k,r['t'],f"{r['weighted_cosine']:.6f}",f"{r['weighted_length_ratio']:.6f}",alias(q['RUN_ID'],'diagnostics '+k)] for k,q in diag.items() for r in q['fields']],'llrrl')
trajectory=table(out,'trajectory','Доля 1000 траекторий DDPM, направленных к итоговой моде.',
 ['$t/T$','Linear','Cosine'],
 [[f,float(diag['linear']['trajectory_modes'][i]['agreement_all']),float(diag['cosine']['trajectory_modes'][i]['agreement_all'])] for i,f in enumerate([.8,.6,.4,.2,.05])],'rrr')
e3rows=rows('E3_models.csv')
e3=table(out,'e3','E3: A --- DDPM T, B --- DDIM 50; общая linear-сеть внутри seed.',
 ['Seed','Arm','Cov',r'$\hat p_0$',r'$D_{mode}$',r'MMD$^2$','Ист.'],
 [[r['seed'],r['arm'],r['Cov'],f"{float(r['rare_frequency']):.4f}",f"{float(r['Dmode']):.5f}",f"{float(r['MMD2']):.8f}",alias(r['RUN_ID'],f"E3 {r['arm']} s{r['seed']}")] for r in e3rows],'llrrrrl')
rare=table(out,'rare','Новый target и полное обучение A при уменьшенном весе редкой моды.',
 [r'$\pi_0$','Train: компонент 0',r'$\hat p_0$','Cov',r'$D_{mode}$','Ист.'],
 [[r['pi0'],r['train_component0_count'],f"{float(r['rare_frequency']):.4f}",r['Cov'],f"{float(r['Dmode']):.5f}",alias(r['RUN_ID'],'rare pi'+r['pi0'])] for r in rows('supplement_rare.csv')],'rrrrrl')
figspec=[
 ('schedules','E1_schedules.png','E1: beta, накопленный сигнал и log SNR двух расписаний.','.37'),
 ('clouds','E1_forward_clouds.png','Прямой процесс при пяти уровнях шума.','.35'),
 ('training','E1_training.png','E1: train и фиксированный val loss при 20 000 обновлений.','.35'),
 ('samples','E1_samples.png','E1: linear и cosine, DDPM K=1000, posterior variance, без clipping.','.36'),
 ('cost','E2_cost_quality.png','E2: качество и время от NFE, linear; DDIM eta=0, DDPM posterior variance.','.43'),
 ('extreme','E2_extreme_K.png','Linear DDIM eta=0, K=1 и 2: сеть и точный предиктор.','.38'),
 ('timestep','diagnostic_timestep.png','Ошибки на 25 шагах для двух расписаний, фиксированные val и шум.','.41'),
 ('scorelinear','diagnostic_score_linear.png','Linear: поля score, косинусы и длины на четырёх уровнях шума.','.53'),
 ('scorecosine','diagnostic_score_cosine.png','Cosine: аналогичная диагностика с весами вероятностной массы.','.53'),
 ('traj','diagnostic_trajectories.png','1000 DDPM-траекторий и постепенное формирование моды.','.35'),
 ('modes','E3_mode_frequencies.png','E3: полные частоты шести мод и внешней категории; linear, DDPM T / DDIM 50 eta=0.','.42'),
 ('pairs','E3_paired_difference.png','Три парные разности Dmode: DDPM минус DDIM 50.','.28'),
 ('rarefig','supplement_rare.png','Дополнительные A-модели с pi0/2 и pi0/4; linear DDPM T.','.32'),
 ('controls','counterfactual_controls.png','Контрфактические оценки: короткое T, десятикратное T и нулевой предиктор.','.30')]
figs={n:figure(out,p/'figures'/f,n,c,h) for n,f,c,h in figspec}
body=r'''\section{Профиль и протокол}
\subsection{Цель и основные соотношения}
Цель --- разделить влияние расписания шума, ошибки сети и численной схемы генерации~\cite{teacher}. При $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s=1}^t\alpha_s$, $\bar\alpha_0=1$ прямой процесс вычисляется за один шаг:
\begin{equation}x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim N(0,I),\quad SNR(t)=\frac{\bar\alpha_t}{1-\bar\alpha_t}.\end{equation}
Линейное расписание интерполирует beta между границами. Косинусное использует $f(t)=\cos^2((t/T+0.008)\pi/(2\cdot1.008))$; после ограничения $\beta_t\le0.999$ произведение $\bar\alpha$ пересчитано. Ядро расписания использует float64.

Сеть предсказывает epsilon; loss --- средний квадрат ошибки по объектам и координатам. Поэтому численные $L_{simple}$ приведены на координату; запись нормы без деления на $d=2$ увеличила бы их вдвое, не меняя обучения. Из предсказания получаются
\begin{equation}\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}},\qquad s_\theta=-\frac{\epsilon_\theta}{\sqrt{1-\bar\alpha_t}}.\end{equation}
Оптимальный предиктор равен $\epsilon_*=E[\epsilon|x_t]$, а не реализованному шуму. Для смеси гауссиан точная плотность после зашумления остаётся смесью с центрами $\sqrt{\bar\alpha_t}\mu_m$ и ковариациями $\bar\alpha_t\Sigma_m+(1-\bar\alpha_t)I$. Score $s_*=\nabla\log p_t$ вычислен стабильно, через log-sum-exp в стандартизованных координатах.

DDPM проходит все $T$ шагов с $\mu_\theta=\alpha_t^{-1/2}(x_t-\beta_t\epsilon_\theta/\sqrt{1-\bar\alpha_t})$ и posterior variance $\tilde\beta_t=\beta_t(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)$. Последний шаг не добавляет шум, clipping выключен. DDIM использует $\tau_i=\operatorname{round}(iT/K)$; при $\eta=0$
\begin{equation}x_{\tau_{i-1}}=\sqrt{\bar\alpha_{\tau_{i-1}}}\hat x_0+\sqrt{1-\bar\alpha_{\tau_{i-1}}}\hat\epsilon_i.\end{equation}
Последний шаг возвращает $\hat x_0$. При полной сетке и $\eta=1$ DDIM совпадает с DDPM. Здесь $\eta$ --- случайность DDIM, не шаг Ланжевена.

\subsection{Данные, сеть и метрики}
Профиль: $T=1000$, $\beta\in[0.0001,0.02]$, $M=6$, $\rho=0.1$, $\pi_0=0.02$. Центры $\mu_m=(\cos(2\pi m/6),\sin(2\pi m/6))$, ковариация $0.01I$. Мода 0 редкая; прочие веса равны 0.196. Train/val/test 20 000/5000/5000, seeds 701/702/703; cal не используется.

Статистики рассчитаны только по train:
\begin{equation}\mu=(-0.18077255,0.00254579),\quad s=(0.62488898,0.77288930).\end{equation} Преобразование по координатам $z=(x-\mu)/s$, обратное $x=z\odot s+\mu$. Модель и точный score работают в $z$, качество считается в исходных единицах.

MLP: вход 2, шесть скрытых слоёв ширины 256, SiLU, синусоидальное вложение времени 128 с отдельным MLP. Преобразованное время вводится в каждый скрытый слой. Adam 0.001, batch 512, 20 000 обновлений; RTX 3060 Laptop, float32, без AMP и TF32. При сравнении расписаний архитектура и все остальные условия одинаковы.

Каждая оценка использует 5000 генерируемых точек и фиксированный test. Частоты цели оценены на 100 000 новых точках, seed 704; внешняя категория включает расстояния не меньше $3\rho$. $Cov$ сравнивает частоту каждой моды с половиной её справочной частоты; $D_{mode}$ --- полная вариация по шести модам и внешней категории. RBF MMD$^2$ --- V-статистика с фиксированной медианной полосой $h=1.23793506$ по всем парам train. Пять свежих истинных выборок задают шум метрик. Числовая полоса и шум приведены далее.

\section{E1. Расписания шума}
Теоретическое ожидание: оба расписания стирают конечный сигнал, но различно распределяют сложность по времени. Доля шагов SNR$<0.01$ равна 0.326 у linear и 0.064 у cosine. Конечный условный KL усреднён по train:
\begin{equation}\overline{KL}_T=\frac12\overline{\bar\alpha_T\|x_0\|^2-d\bar\alpha_T-d\ln(1-\bar\alpha_T)}.\end{equation}
@@E1@@
Оба $\bar\alpha_T$ ниже $10^{-3}$. Cov обеих моделей равен 1. Более низкий linear loss не даёт однозначного превосходства двух метрик качества: разница MMD$^2=-0.000021205$, Dmode=+0.00440.
@@SCHEDULES@@
Для фиксированного $x_0$ получено 10 000 независимых шумовых реализаций на каждом из пяти t. Все координаты прошли исходную проверку четырёх стандартных ошибок; seed не подбирался (@@MOMID@@).
@@MOMENTS@@
@@CLOUDS@@
@@TRAINING@@
@@SAMPLES@@
Граница применимости: $T=100$ при прежних beta даёт $\bar\alpha_T=0.363563$, KL=0.451870, нарушая конечный порог. Это отдельная диагностика точного предиктора, не формальное обучение. Exact DDPM дал MMD$^2=0.000072387$, Dmode=0.01229 (@@SHORTID@@), против 0.000139853 и 0.01627 при полном T. В одном измерении ухудшения нет. Условный KL из формулы не равен KL смеси к нормальному шуму, поэтому наблюдение не отменяет протокольный порог и не доказывает пользу его нарушения.

\section{E2. Качество и стоимость сэмплирования}
Linear-модель E1 заморожена. Меняется только обратная схема; начальные шумы общие, для стохастических шагов зафиксирован отдельный seed. Сеть не переобучается при смене K. Exact --- отдельный предиктор, не обученная сеть.
@@E2@@
@@COST@@
NFE на траекторию равен T у DDPM и K у DDIM. Для $K\ge10$ времена на вызов укладываются в 20\% от соответствующей медианы: максимум отклонения сети 16.37\%, точного предиктора 12.68\%. При полном T точный DDPM и DDIM ниже порога MMD$^2=0.000245055$ (шум плюс три SD). Рост ошибки exact при малых K показывает вклад численной схемы, разрыв network--exact при том же K --- вклад сети.

Для DDIM T использованы три начальных шума. Значения MMD$^2$: 0.005676381, 0.006164185 и 0.004929829; $s_{MMD}=0.000621681$. Критерий
\begin{equation}|MMD^2(K)-MMD^2(T)|\le2s_{MMD}=0.001243363\end{equation}
впервые выполняется на измеренной сетке при $K=50$. Это минимум сетки, а не точный оптимум среди всех целых K.
@@LOW@@
@@EXTREME@@
При K=1 точный DDIM возвращает Tweedie posterior mean, лежащий между модами при почти чистом шуме; Cov=0. При K=2 покрытие также нулевое, а при K=5 появляются покрытые моды: exact Cov=0.5, сеть 0.1667. С K=10 оба предиктора достигают Cov=1, хотя частоты и MMD ещё различаются.

\section{Диагностика сети и траекторий}
Ошибки рассчитаны на общих фиксированных val и шуме в 25 t, включая 1 и T:
\begin{equation}E_\epsilon=\tfrac1dE\|\epsilon-\epsilon_\theta\|^2,\quad E_{x_0}=E_\epsilon/SNR,\quad E_{net}=\tfrac1dE\|\epsilon_*-\epsilon_\theta\|^2.\end{equation}
В ожидании $E_\epsilon=E^*_\epsilon+E_{net}$. На конечной выборке сохраняется измеренный перекрёстный член с SEM; его не обнуляют принудительно. Для exact собственная ошибка равна нулю.
@@TIMESTEP@@
На T linear имеет $E_\epsilon=0.002069320$, $E^*_\epsilon=0.000040065$, $E_{net}=0.002022949$, $E_{x_0}=51.271643$. Cosine: 0.002514774, $2.4111\cdot10^{-9}$, 0.002514743 и $1.035412\cdot10^6$. Разные единицы ошибок принципиальны: маленькая ошибка epsilon при низком SNR может дать большую ошибку восстановления.
@@FIELDS@@
@@SCORELINEAR@@
@@SCORECOSINE@@
Косинусы высоки, но отношение длин не всегда близко к 1: например linear t=800 даёт 1.602079. Вероятностные веса $p_t$ предотвращают преобладание пустых областей над выводом о поле.
@@TRAJECTORY@@
@@TRAJ@@
Начальные шумы фиксированы, для каждой модели сохранено 1000 полных DDPM-траекторий (@@TRAJIDS@@). Ближайший масштабированный центр сравнивается с финальной модой. У linear выбор формируется позднее по индексу t, чем у cosine; индексы соответствуют разным SNR. Рост согласия постепенный, единственный универсальный момент выбора не установлен.

\section{E3. Покрытие редкой моды}
A --- linear DDPM, T=1000, posterior variance; B --- DDIM K=50, eta=0 на тех же весах. Единственный фактор --- конфигурация обратного сэмплера. Фиксированы данные, стандартизация, архитектура, полоса, обучение и общий $x_T$. До дополнительных seed предсказан отрицательный знак $Dmode(A)-Dmode(B)$ из-за дискретизационной ошибки малого K; отсутствие устойчивого отрицательного различия опровергло бы прогноз. Seed 1 явно повторно использует E1/E2, seed 2 и 3 проведены после фиксации гипотезы.
@@E3@@
@@MODES@@
@@PAIRS@@
Три разности $-0.03620,-0.04774,-0.04712$, среднее $-0.043686667$, выборочное SD 0.006491050. Знаки одинаковы, модуль среднего выше $2s_\Delta=0.012982101$ и шума Dmode=0.009434: различие установлено в пользу A, гипотеза поддержана. Все редкие частоты превышают $\pi_0/2=0.01$. Смена seed не переносит вывод на другие цели.
@@RARE@@
@@RAREFIG@@
Для каждого меньшего веса созданы новые train/val/test, заново рассчитаны train-статистики и обучена отдельная A-модель полные 20 000 шагов. При $\pi_0=0.01$ и 0.005 редкая мода остаётся покрытой: $\hat p_0=0.0102$ и 0.0050. Точка потери покрытия в проверенном диапазоне не найдена; неизвестный порог не экстраполируется.

\section{Ответы на контрфактические вопросы}
\subsection*{Вопрос 1. Десятикратное T}
Сохранённое дополнительное обучение при T=10 000 и прежних beta содержит 20 000 обновлений (@@Q1ID@@). $\bar\alpha_T=1.146727\cdot10^{-44}$, средний условный KL такой же по порядку, против $4.035911\cdot10^{-5}$ при T=1000. NFE вырос в 10 раз, медианное время DDPM --- в 11.702753 раза. MMD$^2$ снизилась с 0.002170494 до 0.000369903, Dmode с 0.05625 до 0.02903, Cov=1. Улучшение наблюдалось в одном seed; оно не является необходимым следствием меньшего конечного KL, поскольку меняется и распределение обучающих задач по времени.
\subsection*{Вопрос 2. Нулевой предиктор}
Для $\epsilon_\theta=0$ имеем $v_T=1$, $v_{t-1}=v_t/\alpha_t+\tilde\beta_t$ при $t>1$, $v_0=v_1/\alpha_1$ без последнего шума. Расчёт даёт 49 085.039063, реальные 5000 точек --- дисперсии 49 473.790459 и 49 843.273129, ошибки 0.792\% и 1.545\% (@@Q2ID@@). Без оценки шума обратная цепь расширяет облако и не направляет его к данным.
@@CONTROLS@@
\subsection*{Вопрос 3. Loss и качество}
Linear/cosine val loss 0.408203/0.686859, MMD$^2$ 0.002170494/0.002191699, Dmode 0.05625/0.05185. Linear лучше по численной MMD, cosine по Dmode; различия малы относительно шума. Равномерные t в разных расписаниях дают разные смеси задач: linear имеет 32.6\% низкого SNR, cosine 6.4\%. Высокий шум облегчает epsilon prediction, но не восстановление x0. Поэтому $L_{simple}$ не заменяет качество и не является точным NLL.
\subsection*{Вопрос 4. Усиление терминальной ошибки}
$E_{net}(T)/SNR(T)$ равно 50.122722 для linear и 1 035 399.171539 для cosine. Это MSE относительно точного posterior mean на первом шаге, а не финальная MMD. При DDIM K=5 linear даёт Cov=0.1667, MMD$^2=0.051623$, cosine Cov=0, MMD$^2=0.600008$ (@@COSK5@@). Данные согласуются с сильным усилением ошибки малого SNR; последующие шаги способны частично её исправить, поэтому прямого равенства с финальным качеством нет.
\subsection*{Вопрос 5. Cov недостаточно}
Среди формальных E3 с Cov=1 максимальный Dmode=0.09245 у seed 1, B. Частоты мод 0--5 и внешней категории: 0.0320,0.2366,0.2188,0.1844,0.1512,0.1558,0.0212. Мода 4 недооценена на 0.04349, мода 1 переоценена на 0.04490; редкая мода имеет 0.0320 против справочных 0.01931. Дополнительные траектории этой DDIM-конфигурации дают согласия с финальной модой 0.696,0.712,0.783,0.880,0.981 при t=800,600,400,200,60 (для запроса 50 использован ближайший сохранённый tau=60). Cov допускает дисбаланс и внешние точки; нужны полные частоты и Dmode.

\noindent\textbf{Заключение.} DDPM точнее по частотам, DDIM 50 быстрее.
'''
repl={'E1':e1,'MOMENTS':moments,'E2':e2,'LOW':low,'FIELDS':fields,'TRAJECTORY':trajectory,'E3':e3,'RARE':rare,
 'MOMID':alias(next(iter(d1['theory'].values()))['RUN_ID'],'E1 forward moments'),
 'SHORTID':alias('3500712e1f9546c4fd3a-5b44110051804e37ac2ed9be2c34f09d','T100 exact DDPM'),
 'TRAJIDS':' / '.join(alias(v['trajectory_RUN_ID'],'DDPM trajectory '+k) for k,v in diag.items()),
 'Q1ID':alias(cf[0]['RUN_IDs'][0],'CF1 T10000'),'Q2ID':alias(cf[1]['RUN_IDs'][0],'CF2 zero predictor'),
 'COSK5':alias('62c7f506ef8820f3492a-948a92e6a72f4c02bd54d3d5458ab738','CF4 cosine K5')}
repl.update({k.upper():v for k,v in figs.items()})
for k,v in repl.items():body=body.replace('@@'+k+'@@',v)
body=body.replace('Числовая полоса и шум приведены далее.', 'Фиксированные значения: '+', '.join(k.replace('_',' ')+f' = {v:.8g}' for k,v in metric.items() if isinstance(v,(int,float)) and any(t in k.lower() for t in ['bandwidth','noise','median_distance']))+'.')
assert '@@' not in body
body+=bibliography(7,'Diffusion и Score-Based Generative Models. Noise schedule, оценка score, численный reverse process и computational trade-offs')+registry(out,reg)
(out/'report_body.tex').write_text(body,'utf-8');compile_report(7)
