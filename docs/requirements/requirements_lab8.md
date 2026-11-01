# Lab8: итоговая проверка требований

Источник: свежепрочитанный PDF, редакция 4, 14 страниц. Формальный отчёт по указанию пользователя не создаётся.

**Ограничения:** course split UNVERIFIED; local corruptions are not certified CIFAR-10-C. E2 scientific hypothesis FAIL: обратный вывод установлен; гипотеза сохранена без правки. Исторические Lab4Q2 BLOCKED, Lab5 split UNVERIFIED, Lab5 E2 FAIL, Lab6 E1 FAIL сохранены.

| ID | PDF: исходное требование | Реализация | Проверка | Статус |
|---|---|---|---|---|
| P01 | §1: «Новые модели не обучаются» | Frozen Lab2 checkpoints; zero optimizer updates | checkpoint hashes and passports | PASS |
| P02 | §6: «Разбиение берётся из ЛР0» | Existing splits/cifar10.json; course protocol UNVERIFIED | role-ID allowlists and historical fingerprint | PASS |
| P03 | §6: «cal ... только для конформного предсказания» | No conformal task; cal never accessed | role_data rejects cal | PASS |
| P04 | §7: profile lab_pick arch, corruption, pgd_eps_255 | Existing Lab1 SHA256 function | independent hash slices | PASS |
| T11 | §8.1.1: «сохраните логиты, метки и идентификаторы» | Batched outputs.pt, records JSONL | recompute all metrics offline | PASS |
| T12 | §8.1.2, (8.4)-(8.5): ECE, NLL, Brier | Float64 metrics, bins and empty count | mandatory tests 1-3 | PASS |
| T13 | §8.1.3, (8.6): «Ищите ln T» | Scalar logT optimization with T=1 candidate | fit manifest and val NLL | PASS |
| T14 | §8.1.4: «принимает только объекты val» | Authoritative val ID allowlist | role rejection tests | PASS |
| T21 | §8.2.1: «не менее четырёх искажений» | Five local operators, six configured levels | 30 corruption tests | PASS |
| T22 | §8.2.2: level0 identity, shape, range, label, seed | Per-object deterministic seeds | bit identity tests | PASS |
| T23 | §8.2.3, (8.7): three uncertainty measures | MSP, entropy, margin; higher means uncertain | extreme-probability stability tests | PASS |
| T24 | §8.2.4: «У OOD ... корректности пустое» | Per-object records, OOD correctness null | scan JSONL | PASS |
| T31 | §8.3.1, (8.9)-(8.10): manual FGSM and PGD | Input autograd, per-object worst restart | restart loss oracle | PASS |
| T32 | §8.3.2: normalization first layer | NormalizedModel raw [0,1] wrapper | exact normalized-logit comparison | PASS |
| T33 | §8.3.3: torch.autograd.grad | No parameter gradients or updates | weights/hash/eval tests | PASS |
| T34 | §8.3.4: epsilon, alpha, K, R, random start | Resolved config plus attack records | every-step constraints | PASS |
| T41 | §8.4.1, (8.11)-(8.12): selective risk, AURC | All-object stable ranking, zero risk None | oracle/full/zero coverage tests | PASS |
| T42 | §8.4.2, (8.13): AUROC AUPR FPR95 | Bool OOD-positive flags, no semantic labels | tie-aware ROC and monotone tests | PASS |
| T43 | §8.4.3: three clean selective curves, far OOD | CIFAR10 test versus full SVHN test | offline uncertainty metrics | PASS |
| T44 | §8.4.4: reliability.json includes undefined | One profile per used formal checkpoint | schema and scope audit | PASS |
| H01 | §9: «Перед экспериментом запишите гипотезу» | E1/E3 and Q3 prospective hypotheses | Git timestamp before formal starts | PASS |
| E11 | §9.1: both models before/after temperature | Same fraction and training budget, seed1 | saved-logit tables | PASS |
| E12 | §9.1: 100% same predictions, accuracy delta0 | Objectwise argmax comparison | full test equality | PASS |
| E13 | §9.1: val NLL does not increase | T=1 explicit candidate | per-model fit metadata | PASS |
| E14 | §9.1: ECE M, bin type, empty intervals | All ECE values carry full scheme | serialized metrics audit | PASS |
| E15 | §9.1 boundary: M5, M50, equal count | Offline sensitivity analysis | tables and reliability plots | PASS |
| E21 | §9.2: every corruption, levels0..5, T1/Tstar | CNN seed1 complete five-type grid | all30 condition coverage | PASS |
| E22 | §9.2 main: clean A vs profile level5 B | Three genuine CNN training seeds1,2,3 | checkpoint protocol equality | PASS |
| E23 | §9.2: hypothesis.md Git before runs | Real E2 prereg commit and immutable file | Git blob/ancestor/time checks | PASS |
| E24 | §9.2 (8.14): all three deltas, mean, sample SD | Positive/negative/undetermined teacher rule | independent NumPy paired calculation | PASS |
| E25 | §9.2: level0 matches E1 within1e-6 | Reuse identical clean logits | all metric differences | PASS |
| E26 | §9.2 boundary: first g<=0 and T5 on val5 | Preserve all gains; corrupted val fit | T5 applied to clean and shift test | PASS |
| E31 | §9.3: both models, same test subset | Full shared 10000 test IDs | ID hashes equal all attacks | PASS |
| E32 | §9.3: FGSM epsilon0,1,2,4,8 /255 | Full grid and mean confidence | eps0 matches clean | PASS |
| E33 | §9.3: profile PGD K10 alpha2.5eps/K | Random start R1; no retraining | accuracy PGD<=FGSM checked | PASS |
| E34 | §9.3: perturbation examples and norms | Saved adversarial pixels and scaled deltas | visual QA and Linf constraints | PASS |
| E35 | §9.3 boundary: PGD20, two starts | Same objects, same epsilon | actual run versus PGD10 | PASS |
| Q1 | §10.1: ECE versus NLL/Brier | Own E1 measured example or honest absence | RUN_ID-backed answer | PASS |
| Q2 | §10.2: T5 on clean and corrupted test | T1, Tstar, T5 full metrics | RUN_ID-backed answer | PASS |
| Q3 | §10.3: random sign Gaussian control | Predict before real control, both models | accuracy ordering and norms | PASS |
| Q4 | §10.4: FGSM confidence and ECE T1/Tstar | Same saved adversarial logits | argmax/errors unchanged | PASS |
| Q5 | §10.5: MSP AURC and AUROC T1/Tstar | Actual selective and SVHN metrics | ranking-change counts | PASS |
| A01 | §11 tests1-6 | Six groups plus engineering and old regression | JUnit results | PASS |
| A02 | §12: plots automatic from saved results | Offline artifact generator | independent byte regeneration | PASS |
| A03 | §13: configs, Tstar fit role, attack params, undefined | Final teacher checklist | independent audit | PASS |
| N01 | §12: formal report | User explicitly defers formal report | NOT_APPLICABLE | NOT_APPLICABLE |
| N02 | §14.1: conformal prediction | Optional, not executed; cal untouched | NOT_APPLICABLE | NOT_APPLICABLE |
| N03 | §14.2: near OOD | Optional; excluded11 protocol unavailable | NOT_APPLICABLE | NOT_APPLICABLE |
| N04 | §14.3: ensemble | Optional, not executed | NOT_APPLICABLE | NOT_APPLICABLE |
