from pathlib import Path
from reports.tools.lab8_common import R,read,config
from src.utils.io import write_json,file_hash

def main():
    cfg=config();e1=read(R/'E1.json');kind=cfg['profile']['lab8_corruption']
    text=f'''# Основное сравнение E2: предварительная гипотеза

A: чистый CIFAR-10 test. B: {kind}, сила 5. Единственный изменяемый фактор внутри каждой пары: входное искажение.
Архитектура из персонального профиля: {cfg['profile']['lab8_arch']}. Три настоящих checkpoint из ЛР2, seeds 1,2,3, fraction=.05 (2150 train), 10000 обновлений, last.pt. Контрольные точки не переобучаются.
Фиксированы: каждый checkpoint и его веса; одни и те же 10000 test ID из splits/cifar10.json; train-нормализация; схема ECE M=15, equal_width, пустые интервалы исключены; независимые от seed модели детерминированные искажения (seed=1, per-object ID). Параметры occlusion: доля площади .5, квадрат round(sqrt(.5*32*32)), заливка .5, воспроизводимая случайная позиция.
Для каждого seed собственная Tstar минимизирует NLL только по чистой val (5000 ID). Одна и та же Tstar применяется к A и B. Ни test, ни cal, ни OOD не используются для её подбора. T5 является отдельной диагностикой после основного сравнения.

По (8.4),(8.6),(8.14): g=ECE(T=1)-ECE(Tstar), Delta_s=g_A(s)-g_B(s).
Предсказываем Delta_s>0: температура, подобранная на чистом распределении, уменьшает его ECE сильнее, чем ECE при сильном сдвиге, поскольку валидирующее правдоподобие не оптимизировано для сдвига.
Гипотеза поддержана, только если все три Delta_s положительны и mean(Delta)>=2*sample_sd(Delta). Три отрицательные разности и abs(mean)>=2*sample_sd устанавливают обратный вывод и опровергают прогноз. Во всех остальных случаях: «различие переноса не установлено»; прогноз не подтверждён. Seeds, схему ECE и направление разности после наблюдения менять нельзя.

E1 завершён до этой записи; все три формальные пары E2 ещё не запущены. Чистые seed1 logits из E1 повторно используются с явной ссылкой, не выдаются за новый независимый запуск. Выбор архитектуры/искажения/силы задан профилем и PDF, не подбирался по E2. Источник split курса остаётся UNVERIFIED.
'''
    path=R/'hypothesis.md'
    if path.exists():raise RuntimeError('Never overwrite an existing prospective hypothesis')
    path.write_text(text,encoding='utf-8')
    sources=read(R/'model_sources.json');write_json(R/'E2_protocol_lock.json',{'E1_sha256':file_hash(R/'E1.json'),'hypothesis_sha256':file_hash(path),
        'source_checkpoints':{k:v['sha256'] for k,v in sources.items() if k.startswith('cnn_')},'condition_A':'clean','condition_B':kind+'_5','delta_direction':'gA-gB','expected_sign':'positive','seeds':[1,2,3],
        'ECE':cfg['ece'],'corruption_parameters':cfg['corruptions'][kind],'shared_test_count':10000,'clean_seed1_reuse':'E1 clean_test_cnn_1','Tstar_fit':'each model clean val only'})

if __name__=='__main__':main()
