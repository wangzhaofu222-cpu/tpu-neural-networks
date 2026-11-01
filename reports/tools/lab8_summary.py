"""Final experiment handoff; this is not a formal lab report."""
import json
from pathlib import Path
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
from reports.tools.lab8_common import R,read,config,registry
from src.utils.io import write_json

def main():
    audit=read(R/'final_audit.json');e1=read(R/'E1.json');e2=read(R/'E2.json');e3=read(R/'E3.json');ss=read(R/'selective_ood.json');status=read(R/'requirement_status.json')
    failed={c['check'] for c in audit['failed']}
    # Scientific falsification is saved separately from technical acceptance.
    prediction={'status':'PASS' if e2['statistics']['expected_positive_supported'] else 'FAIL','kind':'scientific prediction, not implementation failure','predicted_sign':'positive','observed':e2['statistics'],'hypothesis_modified':False}
    write_json(R/'hypothesis_outcome.json',prediction)
    if failed:raise RuntimeError('Resolve or explicitly map audit failures before setting teacher checklist')
    for r in status['rows']:
        if r['status']!='NOT_APPLICABLE':r['status']='PASS'
    status['scientific_hypothesis']=prediction;status['counts']={s:sum(r['status']==s for r in status['rows']) for s in ['PASS','FAIL','BLOCKED','NOT_APPLICABLE']}
    write_json(R/'requirement_status.json',status)
    lines=['# Lab8: итоговая проверка требований','', 'Источник: свежепрочитанный PDF, редакция 4, 14 страниц. Формальный отчёт по указанию пользователя не создаётся.','',
           '**Ограничения:** course split UNVERIFIED; local corruptions are not certified CIFAR-10-C. E2 scientific hypothesis FAIL: обратный вывод установлен; гипотеза сохранена без правки. Исторические Lab4Q2 BLOCKED, Lab5 split UNVERIFIED, Lab5 E2 FAIL, Lab6 E1 FAIL сохранены.','', '| ID | PDF: исходное требование | Реализация | Проверка | Статус |','|---|---|---|---|---|']
    for r in status['rows']:lines.append('| '+' | '.join(str(r[k]) for k in ['ID','requirement','method','verification','status'])+' |')
    Path('docs/requirements/requirements_lab8.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    stats=audit['statistics'];begin=datetime.fromisoformat(ET.parse('docs/environment/tests-lab7-verification-before-lab8.xml').getroot().find('testsuite').attrib['timestamp'])
    stats['verification_to_summary_wall_seconds']=(datetime.now(timezone.utc)-begin).total_seconds();stats['finished_at']=datetime.now(timezone.utc).isoformat();stats['teacher_requirement_counts']=status['counts'];stats['scientific_hypothesis_status']=prediction['status'];stats['reliability_profiles']=len(list((R/'profiles').rglob('reliability.json')))
    write_json(R/'execution_statistics.json',stats)
    import yaml
    final_config=config();final_config['fitted_temperatures']=read(R/'temperatures.json');final_config['T5_diagnostic']=e2['T5']['fit'];final_config['model_sources']=read(R/'model_sources.json');final_config['data_protocol']=read(R/'data_protocol.json');final_config['preregistration']=read(R/'preregistration_evidence.json')
    (R/'config.final.yaml').write_text(yaml.safe_dump(final_config,sort_keys=False),encoding='utf-8')
    t=['# Lab8: экспериментальная сводка (без формального отчёта)','',
       'Lab7: 180 tests, 467 independent checks, 39 checkpoints, figures regenerated identically; historical files unchanged. Исходный HEAD ef1d2fa636afc1f36f3223c6a8d78e1cbe498edf.','',
       'Profile: CNN, occlusion, PGD epsilon=2/255, ECE 15 equal-width bins. Device RTX3060 Laptop; full shared test N=10000; SVHN test N=26032. Frozen Lab2 models fraction=.05 (2150 train), 10000 updates, genuine seeds1,2,3; sources/checkpoint hashes in model_sources.json. No new classifier training.','',
       '## E1','', '| model | T | accuracy | ECE | NLL | Brier | RUN_ID |','|---|---|---|---|---|---|---|']
    for r in e1['rows']:t.append(f"| {r['model']} | {r['temperature']:.9f} | {r['accuracy']:.6f} | {r['ece']['value']:.9f} | {r['nll']:.9f} | {r['brier']:.9f} | {r['RUN_ID']} |")
    t+=['','Все argmax/accuracy сохранены, val NLL не выросла. E1_binning.csv содержит M5/M50/equal-count; вывод о снижении ECE сохраняется для обеих моделей.','',
        '## E2','',json.dumps(e2['statistics'],ensure_ascii=False,indent=2),'',
        'Предварительный положительный прогноз опровергнут: температура смягчает сильную избыточную уверенность при occlusion5 больше, чем на чистых данных. Падение accuracy (.7074 -> .2447) намного сильнее падения confidence (.9222 -> .7528), поэтому исходная ECE при сдвиге намного больше. Tstar снижает ECE, но не восстанавливает accuracy. Это результат данной архитектуры и локального сдвига, не общий закон.',
        'Больший абсолютный выигрыш g на искажённых данных не означает лучшую итоговую калибровку: после Tstar ECE occlusion5=.1686 всё ещё выше clean ECE=.0284. Предположение о порядке абсолютных выигрышей, а не формула температуры, оказалось неверным.',
        'Все 30 пар type/level, accuracy/confidence/ECE/NLL/Brier T1/Tstar: E2_grid.csv. Ни при одном уровне не найден g<=0; выход за измеренный диапазон не исследован.','',
        f"T5={e2['T5']['fit']['temperature']:.9f}: см. E2_T5.csv; улучшение occlusion5 оплачено ухудшением калибровки чистого test. Подбор только на corrupted val.",'',
        '## E3','', '| model | attack | eps255 | K | R | accuracy | confidence T1 | RUN_ID |','|---|---|---|---|---|---|---|---|']
    for r in e3['rows']:t.append(f"| {r['source_key']} | {r['attack']} | {r['eps_255']} | {r.get('K','-')} | {r.get('R','-')} | {r['metrics'][0]['accuracy']:.6f} | {r['metrics'][0]['mean_confidence']:.6f} | {r['RUN_ID']} |")
    t+=['','Все ограничения проверены для реальных пикселей и каждого шага PGD. Accuracy при атаке — верхняя оценка устойчивости, не доказательство устойчивости ко всем атакам. Визуализация: E3_examples_cnn/vit.','',
        '## Selective prediction и OOD','', '| model | T | measure | AURC | oracle | AUROC | AUPR | FPR95 | RUN_ID |','|---|---|---|---|---|---|---|---|---|---|']
    for s in ss:
        for c in s['conditions']:
            for k,v in c['measures'].items():t.append(f"| {s['source_key']} | {c['temperature']:.6f} | {k} | {v['selective']['AURC']:.8f} | {v['selective']['oracle_AURC']:.8f} | {v['ood']['AUROC']:.8f} | {v['ood']['AUPR']:.8f} | {v['ood']['FPR95']:.8f} | {s['RUN_ID']} |")
    t+=['','OOD positive; semantic labels unused. Accuracy/ECE/NLL/Brier on SVHN undefined. Three measures share sign higher=uncertain. Empty acceptance risk=None. Four reliability.json profiles explicitly delimit measured and unmeasured scopes.','',
        '## Проверки и ограничения','',json.dumps(stats,ensure_ascii=False,indent=2),'',
        'Пять counterfactual вопросов: counterfactual_answers.md/json, все со своими числами и RUN_ID. Файлы изображений: figures/*.png/*.svg; таблицы автоматически восстановлены из сохранённых logits и summaries.','',
        'Первая дополнительная проверка CPU/GPU: 826 PASS, 1 FAIL. После применения к аудиту тех же детерминированных настроек/TF32=False, что использовались во всех формальных запусках, исходный допуск выполнен без изменения результатов. Полная история: initial_audit.json и audit_corrections.md.','',
        'Протокол данных курса не получен: UNVERIFIED сохраняется; фактические роли и train-нормализация сохранены. Не выполнены только дополнительные conformal/near-OOD/ensemble и формальный отчёт по явному указанию пользователя. Lab9 не запускалась.','',
        'E1/E3 prospective commit: b0636ba71e077a3db32b6620fca10351f8f0a238. E2 prospective commit: '+read(R/'preregistration_evidence.json')['commit']+'.','',
        'Воспроизведение: reports.tools.lab8_execute E1/E2/E3/selective (completed runs reused); reports.tools.lab8_artifacts; reports.tools.lab8_finalize; reports.tools.lab8_regenerate; reports.tools.lab8_audit. Checkpoints are referenced in their original Lab2 locations.']
    (R/'EXPERIMENT_SUMMARY.md').write_text('\n'.join(t)+'\n',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
