# Lab8: экспериментальная сводка (без формального отчёта)

Lab7: 180 tests, 467 independent checks, 39 checkpoints, figures regenerated identically; historical files unchanged. Исходный HEAD ef1d2fa636afc1f36f3223c6a8d78e1cbe498edf.

Profile: CNN, occlusion, PGD epsilon=2/255, ECE 15 equal-width bins. Device RTX3060 Laptop; full shared test N=10000; SVHN test N=26032. Frozen Lab2 models fraction=.05 (2150 train), 10000 updates, genuine seeds1,2,3; sources/checkpoint hashes in model_sources.json. No new classifier training.

## E1

| model | T | accuracy | ECE | NLL | Brier | RUN_ID |
|---|---|---|---|---|---|---|
| cnn | 1.000000000 | 0.707400 | 0.214846305 | 1.640314952 | 0.493711968 | 73968c7a84a5394c6aa0-5b8b87df36624d71a8c08781804ed7fd |
| cnn | 2.956377855 | 0.707400 | 0.028417515 | 0.909764910 | 0.408942834 | 73968c7a84a5394c6aa0-5b8b87df36624d71a8c08781804ed7fd |
| vit | 1.000000000 | 0.450200 | 0.467313358 | 4.065377470 | 0.987853486 | 0799d8eab079cab5e620-ec0903e00a2f4df899b1f979e3032c7c |
| vit | 4.273312679 | 0.450200 | 0.066777192 | 1.674855245 | 0.709218882 | 0799d8eab079cab5e620-ec0903e00a2f4df899b1f979e3032c7c |

Все argmax/accuracy сохранены, val NLL не выросла. E1_binning.csv содержит M5/M50/equal-count; вывод о снижении ECE сохраняется для обеих моделей.

## E2

{
  "deltas": [
    -0.15304490235255033,
    -0.15894211183751564,
    -0.1741639526806952
  ],
  "mean": -0.16205032229025373,
  "sample_sd": 0.010897215279811707,
  "all_positive": false,
  "all_negative": true,
  "abs_mean_ge_2sd": true,
  "conclusion": "обратный вывод установлен",
  "expected_positive_supported": false
}

Предварительный положительный прогноз опровергнут: температура смягчает сильную избыточную уверенность при occlusion5 больше, чем на чистых данных. Падение accuracy (.7074 -> .2447) намного сильнее падения confidence (.9222 -> .7528), поэтому исходная ECE при сдвиге намного больше. Tstar снижает ECE, но не восстанавливает accuracy. Это результат данной архитектуры и локального сдвига, не общий закон.
Больший абсолютный выигрыш g на искажённых данных не означает лучшую итоговую калибровку: после Tstar ECE occlusion5=.1686 всё ещё выше clean ECE=.0284. Предположение о порядке абсолютных выигрышей, а не формула температуры, оказалось неверным.
Все 30 пар type/level, accuracy/confidence/ECE/NLL/Brier T1/Tstar: E2_grid.csv. Ни при одном уровне не найден g<=0; выход за измеренный диапазон не исследован.

T5=6.108733967: см. E2_T5.csv; улучшение occlusion5 оплачено ухудшением калибровки чистого test. Подбор только на corrupted val.

## E3

| model | attack | eps255 | K | R | accuracy | confidence T1 | RUN_ID |
|---|---|---|---|---|---|---|---|
| cnn_1 | FGSM | 0 | 1 | 1 | 0.707400 | 0.922246 | f669bc74fac0f59ecb07-92578798ac0a4ac881373db791f892a9 |
| cnn_1 | FGSM | 1 | 1 | 1 | 0.472100 | 0.919792 | 269235b4e42469f1f427-31afe187d8874ccd90810cd729cb9751 |
| cnn_1 | FGSM | 2 | 1 | 1 | 0.298800 | 0.924621 | 1a44bb9104623fa07cb4-60472984ecd94626b9d910b76e11e7c3 |
| cnn_1 | FGSM | 4 | 1 | 1 | 0.128700 | 0.934521 | 9e73f0a9a788cd0a7e15-633c44687fba46c7b47e051459944848 |
| cnn_1 | FGSM | 8 | 1 | 1 | 0.034400 | 0.938055 | e2b28e527a75a35a8c24-4ddb2e6f24c54581951d73310baa4fbd |
| cnn_1 | PGD | 2.0 | 10 | 1 | 0.245800 | 0.958685 | 00aed6b3019eda8bd2b0-e1833f1367e4430f8aeccd94a49d11b6 |
| cnn_1 | PGD | 2.0 | 20 | 2 | 0.243100 | 0.960176 | 669f74441c1d0bdf9cf6-ef275b8799014ee79fb03a5693b2d6ee |
| cnn_1 | random | 2.0 | - | - | 0.705700 | 0.921821 | 07fec26d48c701023bbd-28ec2dc11eff416185d2bb75ca985d2f |
| vit_1 | FGSM | 0 | 1 | 1 | 0.450200 | 0.917513 | a05b46698998adbde31d-f3ad6d2937eb4c93b6b015c90911a9e0 |
| vit_1 | FGSM | 1 | 1 | 1 | 0.210000 | 0.948671 | 22db9ba6ab867bcc1d57-e727981c4ca647488299d2fa01151396 |
| vit_1 | FGSM | 2 | 1 | 1 | 0.103100 | 0.961179 | be749c9af8d7ccb534e2-69374eeafcc54854810477dd4e5d9500 |
| vit_1 | FGSM | 4 | 1 | 1 | 0.040300 | 0.966628 | 82a6087687cc2c8c5c9b-19454af7b1184b9299a761b6bea4cf27 |
| vit_1 | FGSM | 8 | 1 | 1 | 0.017400 | 0.956018 | 8d1e5819f7bc0d28b6dd-c705a88f9353488ab16a0427fcb193ef |
| vit_1 | PGD | 2.0 | 10 | 1 | 0.076000 | 0.985781 | 79a1b4800fce5956a2d9-2d085f1bce3b439aa37201e020dabd75 |
| vit_1 | PGD | 2.0 | 20 | 2 | 0.075400 | 0.986183 | f43e239aca6b8f9a3c27-6d81a2d31c5a46c59dd4c858023ddcfd |
| vit_1 | random | 2.0 | - | - | 0.448900 | 0.917750 | 4ed189b2245186f3b436-7011dd2f77284357931c0e901a28d9d2 |

Все ограничения проверены для реальных пикселей и каждого шага PGD. Accuracy при атаке — верхняя оценка устойчивости, не доказательство устойчивости ко всем атакам. Визуализация: E3_examples_cnn/vit.

## Selective prediction и OOD

| model | T | measure | AURC | oracle | AUROC | AUPR | FPR95 | RUN_ID |
|---|---|---|---|---|---|---|---|---|---|
| cnn_1 | 1.000000 | msp | 0.11593428 | 0.04774175 | 0.67409906 | 0.80419183 | 0.70370000 | c4bd07267e884795ad42-9f79f9e6c4534bf9ae92ce4d5fbd849f |
| cnn_1 | 1.000000 | entropy | 0.11543846 | 0.04774175 | 0.67300327 | 0.79990149 | 0.70340000 | c4bd07267e884795ad42-9f79f9e6c4534bf9ae92ce4d5fbd849f |
| cnn_1 | 1.000000 | margin | 0.11639981 | 0.04774175 | 0.67435897 | 0.80666093 | 0.70420000 | c4bd07267e884795ad42-9f79f9e6c4534bf9ae92ce4d5fbd849f |
| cnn_1 | 2.956378 | msp | 0.11497476 | 0.04774175 | 0.66730591 | 0.79346952 | 0.70570000 | c4bd07267e884795ad42-9f79f9e6c4534bf9ae92ce4d5fbd849f |
| cnn_1 | 2.956378 | entropy | 0.11641951 | 0.04774175 | 0.65571562 | 0.77766120 | 0.70920000 | c4bd07267e884795ad42-9f79f9e6c4534bf9ae92ce4d5fbd849f |
| cnn_1 | 2.956378 | margin | 0.11551804 | 0.04774175 | 0.67235874 | 0.80468133 | 0.70470000 | c4bd07267e884795ad42-9f79f9e6c4534bf9ae92ce4d5fbd849f |
| vit_1 | 1.000000 | msp | 0.39306385 | 0.19053937 | 0.57893279 | 0.75999314 | 0.86400000 | 28c6a867fdd60b50e742-b8ff8463a4ac44f38517e4b96fe64ebe |
| vit_1 | 1.000000 | entropy | 0.39192767 | 0.19053937 | 0.58026659 | 0.76186127 | 0.86320000 | 28c6a867fdd60b50e742-b8ff8463a4ac44f38517e4b96fe64ebe |
| vit_1 | 1.000000 | margin | 0.39472028 | 0.19053937 | 0.57787625 | 0.75914036 | 0.86580000 | 28c6a867fdd60b50e742-b8ff8463a4ac44f38517e4b96fe64ebe |
| vit_1 | 4.273313 | msp | 0.38504063 | 0.19053937 | 0.59200607 | 0.76797288 | 0.85300000 | 28c6a867fdd60b50e742-b8ff8463a4ac44f38517e4b96fe64ebe |
| vit_1 | 4.273313 | entropy | 0.38249034 | 0.19053937 | 0.60249693 | 0.77292771 | 0.84800000 | 28c6a867fdd60b50e742-b8ff8463a4ac44f38517e4b96fe64ebe |
| vit_1 | 4.273313 | margin | 0.39008687 | 0.19053937 | 0.58336211 | 0.76164987 | 0.85720000 | 28c6a867fdd60b50e742-b8ff8463a4ac44f38517e4b96fe64ebe |
| cnn_2 | 1.000000 | msp | 0.12510975 | 0.05457720 | 0.66005669 | 0.79814789 | 0.73060000 | ee51a112fbd8ca02ceac-0fbadfbec7b946c6b6e26b3bedc5b2a1 |
| cnn_2 | 1.000000 | entropy | 0.12452940 | 0.05457720 | 0.66002131 | 0.79573217 | 0.72990000 | ee51a112fbd8ca02ceac-0fbadfbec7b946c6b6e26b3bedc5b2a1 |
| cnn_2 | 1.000000 | margin | 0.12571559 | 0.05457720 | 0.65970869 | 0.79933001 | 0.73250000 | ee51a112fbd8ca02ceac-0fbadfbec7b946c6b6e26b3bedc5b2a1 |
| cnn_2 | 3.103821 | msp | 0.12270162 | 0.05457720 | 0.65947909 | 0.79380090 | 0.72860000 | ee51a112fbd8ca02ceac-0fbadfbec7b946c6b6e26b3bedc5b2a1 |
| cnn_2 | 3.103821 | entropy | 0.12355193 | 0.05457720 | 0.65481016 | 0.78722322 | 0.73080000 | ee51a112fbd8ca02ceac-0fbadfbec7b946c6b6e26b3bedc5b2a1 |
| cnn_2 | 3.103821 | margin | 0.12404161 | 0.05457720 | 0.66012855 | 0.79909927 | 0.72790000 | ee51a112fbd8ca02ceac-0fbadfbec7b946c6b6e26b3bedc5b2a1 |
| cnn_3 | 1.000000 | msp | 0.11108611 | 0.04606226 | 0.63955390 | 0.78885714 | 0.76230000 | 62769d38becb70261a95-0f87c25adf8344b599bf1396638e8f47 |
| cnn_3 | 1.000000 | entropy | 0.11072556 | 0.04606226 | 0.63964453 | 0.78916814 | 0.76220000 | 62769d38becb70261a95-0f87c25adf8344b599bf1396638e8f47 |
| cnn_3 | 1.000000 | margin | 0.11152248 | 0.04606226 | 0.63907052 | 0.78731115 | 0.76190000 | 62769d38becb70261a95-0f87c25adf8344b599bf1396638e8f47 |
| cnn_3 | 3.114989 | msp | 0.11003919 | 0.04606226 | 0.63784261 | 0.78643025 | 0.75700000 | 62769d38becb70261a95-0f87c25adf8344b599bf1396638e8f47 |
| cnn_3 | 3.114989 | entropy | 0.11168526 | 0.04606226 | 0.63195009 | 0.78200621 | 0.75770000 | 62769d38becb70261a95-0f87c25adf8344b599bf1396638e8f47 |
| cnn_3 | 3.114989 | margin | 0.11053875 | 0.04606226 | 0.63924543 | 0.78670777 | 0.75800000 | 62769d38becb70261a95-0f87c25adf8344b599bf1396638e8f47 |

OOD positive; semantic labels unused. Accuracy/ECE/NLL/Brier on SVHN undefined. Three measures share sign higher=uncertain. Empty acceptance risk=None. Four reliability.json profiles explicitly delimit measured and unmeasured scopes.

## Проверки и ограничения

{
  "tests_passed": 239,
  "Lab7_audit_checks": 467,
  "checks": 827,
  "formal_evaluation_runs": 65,
  "smoke_runs": 2,
  "classifier_training_runs": 0,
  "new_checkpoints": 0,
  "source_checkpoints_verified": 6,
  "formal_run_seconds_sum": 1214.6391196999975,
  "audit_seconds": 68.92671239999981,
  "all_run_folders": 67,
  "figure_groups": 20,
  "verification_to_summary_wall_seconds": 2445.371047,
  "finished_at": "2026-11-01T05:47:39.114346+00:00",
  "teacher_requirement_counts": {
    "PASS": 45,
    "FAIL": 0,
    "BLOCKED": 0,
    "NOT_APPLICABLE": 4
  },
  "scientific_hypothesis_status": "FAIL",
  "reliability_profiles": 4
}

Пять counterfactual вопросов: counterfactual_answers.md/json, все со своими числами и RUN_ID. Файлы изображений: figures/*.png/*.svg; таблицы автоматически восстановлены из сохранённых logits и summaries.

Первая дополнительная проверка CPU/GPU: 826 PASS, 1 FAIL. После применения к аудиту тех же детерминированных настроек/TF32=False, что использовались во всех формальных запусках, исходный допуск выполнен без изменения результатов. Полная история: initial_audit.json и audit_corrections.md.

Протокол данных курса не получен: UNVERIFIED сохраняется; фактические роли и train-нормализация сохранены. Не выполнены только дополнительные conformal/near-OOD/ensemble и формальный отчёт по явному указанию пользователя. Lab9 не запускалась.

E1/E3 prospective commit: b0636ba71e077a3db32b6620fca10351f8f0a238. E2 prospective commit: a85340648bdc9bfe42fc6f2a92732b03182ec6f8.

Воспроизведение: reports.tools.lab8_execute E1/E2/E3/selective (completed runs reused); reports.tools.lab8_artifacts; reports.tools.lab8_finalize; reports.tools.lab8_regenerate; reports.tools.lab8_audit. Checkpoints are referenced in their original Lab2 locations.
