"""Write the immutable E3 hypothesis only after E1/E2 exist; Git commit is a separate explicit step."""
from datetime import datetime,timezone
from pathlib import Path
import json
from src.utils.io import write_json,file_hash

ROOT=Path('experiments/lab4')
def main():
    if (ROOT/'hypothesis.md').exists():raise RuntimeError('Refusing rewrite of an existing preregistered hypothesis')
    result=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
    if 'E1' not in result or 'E2' not in result or len(result['E2']['rows'])!=6:raise RuntimeError('E1 andall6E2conditions mustbefinished first')
    profile=json.loads((ROOT/'profile.json').read_text());sources=json.loads((ROOT/'E1_sources.json').read_text())
    g1=result['E1']['G1'];prediction=.5*g1 if abs(g1)>=.01 else .05;sign=1 if prediction>0 else -1
    timestamp=datetime.now(timezone.utc).isoformat();rmin=profile['rmin'];fractions=profile['lab4_label_fractions']
    snapshot=ROOT/'E1_E2_before_E3.json'
    if snapshot.exists():raise RuntimeError('Refusing overwrite of the pre-E3 result snapshot')
    snapshot.write_bytes((ROOT/'results.json').read_bytes())
    feature_records=json.loads((ROOT/'features.json').read_text(encoding='utf-8'))
    encoders={}
    for name,label in [('B0','E1_B0_features'),('B1','E1_B1_features'),('SimCLR','E1_SSL_epoch100_features')]:
        feature=next(r for r in feature_records if r['label']==label)
        checkpoint=Path(sources[name])/'checkpoints'/feature['checkpoint']
        assert file_hash(checkpoint)==feature['checkpoint_sha256']
        encoders[name]={'path':sources[name],'RUN_ID':Path(sources[name]).name,
            'checkpoint':feature['checkpoint'],'checkpoint_sha256':feature['checkpoint_sha256'],
            'feature_RUN_ID':feature['RUN_ID'],'encoder_state_sha256':feature['encoder_state_sha256']}
    selection={'registered_at_utc':timestamp,'A':'SimCLR','B':'B0','rmin':rmin,'fractions':fractions,
        'predicted_sign':sign,'predicted_mean_difference':prediction,'prediction_basis':'half of the measured E1full-label gain as aheuristicpointforecast; ifnearzero, +.05theoreticalexploratoryforecast',
        'E1_G1':g1,'encoders':encoders,'E1_E2_results_sha256':file_hash(ROOT/'results.json'),
        'E1_E2_snapshot_path':snapshot.as_posix(),
        'probe_seeds':[1,2,3],'pretraining_seed':1,'unit_of_analysis':'image',
        'reuse':'The3seed1/r1 cells reuse exactlyidentical E1probes; all9rmincells andother24nonreusedE3cells are unseen. Reuse isexplicit, notrepresented as newtraining.',
        'falsification':'Anestablished opposite sign contradicts direction; mixed/zero signs orabsmean<2SD mean differenceunestablished. Pointforecast error isreported separately, no newmargin replaces teachercriterion.'}
    write_json(ROOT/'E3_selection.json',selection)
    text=f'''# E3: гипотеза до новых запусков

Дата UTC: {timestamp}.

A — SimCLR из E1, B — случайный B0 из E1. Меняется только сохранённый кодировщик; предобучение повторно не выполняется. B1 также оценивается во всей матрице как ориентир, уже использовавший все метки train.

Фиксировано: CIFAR-10, исходное разбиение ЛР0 (43000/5000/2000/10000), замороженные признаки h размерности512, eval/noaugmentation при извлечении. Cal не читается. Основная доля rmin={rmin:g} (860изображений,86на класс); полная сетка {fractions}. Кодировщики имеют одинаковую архитектуру CIFARResNet18; их RUN_ID и SHA256 зафиксированы в E3_selection.json.

Для seeds1,2,3 строятся стратифицированные вложенные подвыборки: отдельный NumPy PCG64(seed) переставляет изображения каждого класса train; для каждого r берётся префикс. Все три кодировщика используют одинаковые индексы при данном(seed,r). Приr1 используется весьtrain. Инициализация probe использует Torch seed с темже номером, но отдельным состоянием RNG; аугментации и разбиение не потребляют этот генератор.

Общий probe: линейный слой с bias, полная batch-оптимизация SGD(momentum=.9,lr=.1),200эпох, cosine без warmup. Стандартизация каждого столбца признаков вычисляется только на выбранной train-подвыборке, stdfloor1e-5. L2 из[0,.001,.01] выбирается по максимальной итоговой accuracy на5000val, при равенстве меньшееL2. Test используется только для окончательной accuracy. Кодировщик, его BatchNorm-буферы и исходный checkpoint не изменяются.

E1далG(1)={g1:.9f}. Численная эвристическая точечная гипотеза приrmin: среднееΔ=M(SimCLR)-M(B0) около {prediction:+.9f}; ожидаемый знак {'положительный' if sign>0 else 'отрицательный'}. Прогноз не является доверительным интервалом и не заменяет критерий преподавателя. Ошибка точечного прогноза будет показана отдельно.

Источник случайности — выбор размеченных изображений и начальные веса probe; анализируемая единица — изображение. Один и тот же уже обученный SimCLR используется при всехseed. Поэтому разброс не включает случайность предобучения и не переносится автоматически на метод SimCLR вообще. Выбор регуляризации по5000размеченнымval делает условия малой разметки идеализированными одинаково для всех кодировщиков.

Критерий: публикуются все триΔ, среднее и выборочноеSD(ddof1). Различие установлено только если все триΔ ненулевые одного знака и|mean|>=2SD. Установленный противоположный знак опровергает направление. Смешанные/нулевые знаки либо|mean|<2SD означают «различие не установлено» и отсутствие поддержки направленной гипотезы; seed не заменяются.

Матрица содержит36клеток:3кодировщика×3seeds×4доли. Три клетки(seed1,r1) точно переиспользуют уже необходимые E1probe при техже условиях, без лишнего повторения обучения. Они не выдаются за новые независимые от E1 наблюдения. Все основные rminprobe и остальные24новые клетки ещё не запускались. Существующие E1/E2результаты использованы только для этой заранее записанной гипотезы; её текст после новых результатов не редактируется.
'''
    (ROOT/'hypothesis.md').write_text(text,encoding='utf-8')
    print(json.dumps({'rmin':rmin,'prediction':prediction,'encoders':{k:v['RUN_ID'] for k,v in encoders.items()}},indent=2))
if __name__=='__main__':main()
