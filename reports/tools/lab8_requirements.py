"""Traceable checklist from the freshly reread teacher PDF (14 pages)."""
from pathlib import Path
from src.utils.io import write_json,file_hash

ROWS=[
('P01','§1: «Новые модели не обучаются»','Frozen Lab2 checkpoints; zero optimizer updates','checkpoint hashes and passports'),
('P02','§6: «Разбиение берётся из ЛР0»','Existing splits/cifar10.json; course protocol UNVERIFIED','role-ID allowlists and historical fingerprint'),
('P03','§6: «cal ... только для конформного предсказания»','No conformal task; cal never accessed','role_data rejects cal'),
('P04','§7: profile lab_pick arch, corruption, pgd_eps_255','Existing Lab1 SHA256 function','independent hash slices'),
('T11','§8.1.1: «сохраните логиты, метки и идентификаторы»','Batched outputs.pt, records JSONL','recompute all metrics offline'),
('T12','§8.1.2, (8.4)-(8.5): ECE, NLL, Brier','Float64 metrics, bins and empty count','mandatory tests 1-3'),
('T13','§8.1.3, (8.6): «Ищите ln T»','Scalar logT optimization with T=1 candidate','fit manifest and val NLL'),
('T14','§8.1.4: «принимает только объекты val»','Authoritative val ID allowlist','role rejection tests'),
('T21','§8.2.1: «не менее четырёх искажений»','Five local operators, six configured levels','30 corruption tests'),
('T22','§8.2.2: level0 identity, shape, range, label, seed','Per-object deterministic seeds','bit identity tests'),
('T23','§8.2.3, (8.7): three uncertainty measures','MSP, entropy, margin; higher means uncertain','extreme-probability stability tests'),
('T24','§8.2.4: «У OOD ... корректности пустое»','Per-object records, OOD correctness null','scan JSONL'),
('T31','§8.3.1, (8.9)-(8.10): manual FGSM and PGD','Input autograd, per-object worst restart','restart loss oracle'),
('T32','§8.3.2: normalization first layer','NormalizedModel raw [0,1] wrapper','exact normalized-logit comparison'),
('T33','§8.3.3: torch.autograd.grad','No parameter gradients or updates','weights/hash/eval tests'),
('T34','§8.3.4: epsilon, alpha, K, R, random start','Resolved config plus attack records','every-step constraints'),
('T41','§8.4.1, (8.11)-(8.12): selective risk, AURC','All-object stable ranking, zero risk None','oracle/full/zero coverage tests'),
('T42','§8.4.2, (8.13): AUROC AUPR FPR95','Bool OOD-positive flags, no semantic labels','tie-aware ROC and monotone tests'),
('T43','§8.4.3: three clean selective curves, far OOD','CIFAR10 test versus full SVHN test','offline uncertainty metrics'),
('T44','§8.4.4: reliability.json includes undefined','One profile per used formal checkpoint','schema and scope audit'),
('H01','§9: «Перед экспериментом запишите гипотезу»','E1/E3 and Q3 prospective hypotheses','Git timestamp before formal starts'),
('E11','§9.1: both models before/after temperature','Same fraction and training budget, seed1','saved-logit tables'),
('E12','§9.1: 100% same predictions, accuracy delta0','Objectwise argmax comparison','full test equality'),
('E13','§9.1: val NLL does not increase','T=1 explicit candidate','per-model fit metadata'),
('E14','§9.1: ECE M, bin type, empty intervals','All ECE values carry full scheme','serialized metrics audit'),
('E15','§9.1 boundary: M5, M50, equal count','Offline sensitivity analysis','tables and reliability plots'),
('E21','§9.2: every corruption, levels0..5, T1/Tstar','CNN seed1 complete five-type grid','all30 condition coverage'),
('E22','§9.2 main: clean A vs profile level5 B','Three genuine CNN training seeds1,2,3','checkpoint protocol equality'),
('E23','§9.2: hypothesis.md Git before runs','Real E2 prereg commit and immutable file','Git blob/ancestor/time checks'),
('E24','§9.2 (8.14): all three deltas, mean, sample SD','Positive/negative/undetermined teacher rule','independent NumPy paired calculation'),
('E25','§9.2: level0 matches E1 within1e-6','Reuse identical clean logits','all metric differences'),
('E26','§9.2 boundary: first g<=0 and T5 on val5','Preserve all gains; corrupted val fit','T5 applied to clean and shift test'),
('E31','§9.3: both models, same test subset','Full shared 10000 test IDs','ID hashes equal all attacks'),
('E32','§9.3: FGSM epsilon0,1,2,4,8 /255','Full grid and mean confidence','eps0 matches clean'),
('E33','§9.3: profile PGD K10 alpha2.5eps/K','Random start R1; no retraining','accuracy PGD<=FGSM checked'),
('E34','§9.3: perturbation examples and norms','Saved adversarial pixels and scaled deltas','visual QA and Linf constraints'),
('E35','§9.3 boundary: PGD20, two starts','Same objects, same epsilon','actual run versus PGD10'),
('Q1','§10.1: ECE versus NLL/Brier','Own E1 measured example or honest absence','RUN_ID-backed answer'),
('Q2','§10.2: T5 on clean and corrupted test','T1, Tstar, T5 full metrics','RUN_ID-backed answer'),
('Q3','§10.3: random sign Gaussian control','Predict before real control, both models','accuracy ordering and norms'),
('Q4','§10.4: FGSM confidence and ECE T1/Tstar','Same saved adversarial logits','argmax/errors unchanged'),
('Q5','§10.5: MSP AURC and AUROC T1/Tstar','Actual selective and SVHN metrics','ranking-change counts'),
('A01','§11 tests1-6','Six groups plus engineering and old regression','JUnit results'),
('A02','§12: plots automatic from saved results','Offline artifact generator','independent byte regeneration'),
('A03','§13: configs, Tstar fit role, attack params, undefined','Final teacher checklist','independent audit'),
('N01','§12: formal report','User explicitly defers formal report','NOT_APPLICABLE'),
('N02','§14.1: conformal prediction','Optional, not executed; cal untouched','NOT_APPLICABLE'),
('N03','§14.2: near OOD','Optional; excluded11 protocol unavailable','NOT_APPLICABLE'),
('N04','§14.3: ensemble','Optional, not executed','NOT_APPLICABLE'),
]

def main():
    p=Path('docs/requirements/requirements_lab8.md');p.parent.mkdir(parents=True,exist_ok=True)
    lines=['# Lab 8: требования исходного PDF','', 'Источник: lab8_robustness_calibration_uncertainty_1010.pdf, редакция 4, 14 страниц; заново прочитан полностью, формулы визуально сверены.',
           'Приоритет: PDF определяет эксперименты; пользователь явно исключил формальный отчёт и дополнительные задания. Локальные искажения заданы явно, эквивалентность CIFAR-10-C не утверждается. Протокол курса остаётся UNVERIFIED.','',
           '| ID | PDF: исходное требование | Реализация | Проверка | Статус |','|---|---|---|---|---|']
    states=[]
    for i,req,method,check in ROWS:
        status='NOT_APPLICABLE' if i.startswith('N') else 'IN_PROGRESS'
        lines.append(f'| {i} | {req} | {method} | {check} | {status} |');states.append({'ID':i,'requirement':req,'method':method,'verification':check,'status':status})
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json('experiments/lab8/requirement_status.json',{'rows':states,'PDF_sha256':file_hash('drive-download-20261026T053112Z-1-001/lab8_robustness_calibration_uncertainty_1010.pdf'),'historical_states':{'Lab4Q2':'BLOCKED','Lab5_course_split':'UNVERIFIED','Lab5_E2_magnitude':'FAIL','Lab6_E1_mixing':'FAIL'}})

if __name__=='__main__':main()
