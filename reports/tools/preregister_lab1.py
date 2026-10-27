"""Write an E3 proposal after E1/E2; committing it is a separate explicit step."""
import json
from pathlib import Path
import yaml
from src.config import load_config
from src.utils.io import digest, write_json


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite-root',default='experiments/lab1')
    root=Path(parser.parse_args().suite_root)
    ledger=json.loads((root/'ledger.json').read_text(encoding='utf-8'))
    assert not any(r['label'].startswith('E3_') for r in ledger), 'Cannot preregister after E3'
    expected=[f'E2_{a}_{d}' for a in ['plain','residual_unit','residual_scaled'] for d in [8,24,48,96]]
    assert all(any(r['label']==l for r in ledger) for l in expected)
    current={r['label']:r for r in ledger}
    lr=json.loads((root/'selected_lr.json').read_text())['lr']
    labels=['E2_plain_96','E2_residual_scaled_96']
    observations=[json.loads((Path(current[l]['path'])/'summary.json').read_text())['trainability']['rho_T'] for l in labels]
    signal=json.loads((Path(current[labels[1]]['path'])/'probe.initial.json').read_text())
    import math
    base=load_config('configs/lab1.yaml')
    proposal={'A':{'name':'deep_mlp','depth':96,'alpha':'inverse_sqrt_depth'},
              'B':{'name':'residual_mlp','depth':96,'alpha':'inverse_sqrt_depth'},
              'description_A':'обычная MLP, 96 скрытых слоёв',
              'description_B':'остаточная MLP, 96 блоков, alpha=1/sqrt(96)',
              'factor':'архитектура (model.name); остальные поля модели одинаковы',
              'metric':'rho_T; lower is better','paired_difference':'A minus B','seeds':[1,2,3],
              'epochs':3,'lr':lr,'expected_sign':1,'falsification':'not all deltas positive OR mean(delta)<2*sample_std(delta,ddof=1)',
              'selection_evidence':[{'label':l,'RUN_ID':current[l]['RUN_ID'],'rho_T':rho} for l,rho in zip(labels,observations)],
              'base_config_hash':digest(base),'limitations':'one fixed split, 3 epochs; architecture changes parameter count and number of linear transformations'}
    proposal['known_E2_threshold_exception']={'q_ratio':signal['q_ratio'],'first_block_c':signal['c'],
                 'bound':1.2*math.exp(signal['c']),'status':'FAIL; not hidden or replaced; per-block moment diagnosis planned'}
    write_json(root/'e3_preregistration.json',proposal)
    hypothesis=f'''# E3: априорная гипотеза ЛР1

Этот текст записан после E1/E2, но до всех запусков E3. Числа ниже принадлежат только поисковой E2, не E3.

## Сравнение и единственный фактор

A: обычная MLP, 96 скрытых слоёв. B: остаточная MLP, 96 блоков с alpha=1/sqrt(96). Изменяется только `model.name`. Остальные поля конфигурации модели совпадают. Архитектура включает количество линейных преобразований: A имеет 97 слоёв с классификатором, B — 194 с проекцией и классификатором. Число параметров не уравнивается; причинный вывод не отделяет skip от этого различия бюджета.

## Зафиксированные условия

Профиль P3adf02; CIFAR-10, существующее разбиение ЛР0; полный train 43000; val5000, cal2000, test10000; test не используется. Нормализация только по train. Ширина256, He, ReLU, без нормализации слоёв, нулевые начальные bias. SGD momentum0.9; LR={lr}, weight decay0; постоянная LR; batch128; 3 полные эпохи; float32, без AMP, клиппинга и изменения ширины. Общие буквальные seeds: **1,2,3**. Пробный batch64, rho на фиксированных1024 объектах train; измерения градиента на первом/среднем/последнем batch эпохи. Порядок: A seeds1–3, затем B seeds1–3. Используется общий ExperimentRunner ЛР0.

## Ожидаемая разность

Основная метрика m=rho_T, меньше лучше. Delta_s = m(A,s)−m(B,s). Ожидается, что все три Delta_s положительны и средняя разность не меньше 2s_Delta (выборочное std, ddof=1). Обоснование: (1.13) ограничивает накопление энергии при масштабе ветви 1/sqrt(L), а длинное последовательное произведение обычной сети ухудшает оптимизацию. В E2 на L96 наблюдались rho(A)={observations[0]:.9f}, rho(B)={observations[1]:.9f}; источники: {current[labels[0]]['RUN_ID']}, {current[labels[1]]['RUN_ID']}.

Гипотеза не утверждает достижения rho<=0.5 и не обещает конкретную val accuracy. Факт лучшего rho не тождественен прохождению порога обучаемости.

Известное до E3 исключение: в E2 на L96 измерено qL/q0={signal['q_ratio']:.9f}, что выше границы 1.2 exp(c0)={1.2*math.exp(signal['c']):.9f} при c0={signal['c']:.9f}. Этот FAIL не заменяется другим seed; отдельно проверяются неодинаковые c_l и ненулевые эмпирические перекрёстные члены. Гипотеза E3 относится к rho, а не к выполнению этой численной границы.

## Опровержение и правило вывода

Если хотя бы одна разность неположительна или mean(Delta)<2*std(Delta,ddof=1), положительная гипотеза не подтверждена. Если не выполнено общее правило одинакового знака и abs(mean)>=2std, формулировка: «различие не установлено». Публикуются все шесть запусков и все три разности, включая нежелательные исходы; отсутствующие/нечисленные метрики не заменяются числами. При численном сбое это явно сообщается как ограничение статистического сравнения.

Три seeds характеризуют обучение на одном фиксированном разбиении. Вывод не переносится автоматически на новые данные, другую глубину, другой бюджет параметров, LR или число эпох. После просмотра E3 эта гипотеза не изменяется.
'''
    Path('hypothesis.md').write_text(hypothesis,encoding='utf-8')
    print(json.dumps(proposal,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
