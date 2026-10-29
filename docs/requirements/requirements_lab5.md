# Lab 5 requirements — original PDF revision 4, 14 pages

Scope: reuse Lab0 ExperimentRunner; no formal Russian/LaTeX/PDF report; stop before Lab6. Profile: Cora, GAT, depths [2,3,6,12,24]. User explicitly authorized PyG public masks on 2026-10-29; teacher course split remains independently UNVERIFIED, not falsely certified.

| ID | PDF source / required or optional | Requirement | Implementation / evidence | Verification | Status |
|---|---|---|---|---|---|
| A01 | User prerequisite / required | Verify Lab4 commit709e39c,106tests,334checkpoints,E1–E3,prereg57f3fd3; preserve Q2 BLOCKED and history | experiments/lab4_verification | independent regeneration, checkpoint hashes and regression XML | PASS |
| P01 | §7 / required | SHA256(student_id:lab5), dataset Cora, GAT, [2,3,6,12,24]; reuse profile generator | configs/lab5.yaml; profile.json | independent hash slices | PASS |
| D01 | §6,§8.1 / required | PyG Planetoid public masks; teacher split unavailable; explicit user-authorized substitute, not independent teacher verification | src/graph/datasets.py; split_decision.json | raw test.index, per-role indices/hash/classes, masks disjoint | PASS_SUBSTITUTION; COURSE_SPLIT_UNVERIFIED |
| D02 | §6 / required | Transductive X/edges visible for all nodes; loss train only; val selects config/epoch/depth; no test choice; cal forbidden | shared role loaders/objective | test-label mutation leaves weights/choices identical | PASS |
| D03 | §5.1,§6 / required | Central canonical undirected graph, two directions, no duplicates/self-loops; actual N/E/F/C/counts | src/graph/stats.py | data integrity and expected Cora table comparison | PASS |
| M01 | §8.1 / required | Common model(x,edge_index), forward_features list per layer; MLP ignores edges | src/models/graph | empty/random/original equality | PASS |
| M02 | (5.4),§8.1 / required | GCNConv, self-loops only in layer, symmetric degree normalization, isolated nodes, no output ReLU | src/models/graph | empty GCN equals weight-matched MLP <=1e-6 | PASS |
| M03 | (5.6),§8.1 / required | Assigned GAT: hidden 8heads ×8 features, output head averaging; attention sums one in eval | src/models/graph | attention and shape tests | PASS |
| M04 | §8.1 / required | Depth2,width64,dropout.5,Adam,max200; same lr{.01,.005} × wd{.0005,.005} per model; best val epoch/config only | shared ExperimentRunner | all12 search runs, locked selection before test | PASS |
| M05 | §8.1 reference / required | Separate GCN width16,dropout.5,lr.01,wd.0005; Cora reference81.5%, abs gap>3pp investigation, not test tuning | reference config/run | actual fixed reference test result | PASS |
| S01 | (5.1),(5.7),(5.8),§8.2 / required | N,E,mean/median/max degree,components,isolates,class sizes,hE,h0,hdeg; hE n/a if E0 | src/graph/stats.py | direct formula oracle and known graphs | PASS |
| S02 | §8.2 / required | Every training/evaluation graph statistics and structure hash in RUN metadata | graph manifests | regenerate graph/hash/statistics | PASS |
| G01 | (5.9),§8.3 / required | Independent Bernoulli deletion per undirected edge; both records together; p0 identity,p1 empty | src/graph/perturb.py | six p values, symmetry and reproducibility tests | PASS |
| G02 | §8.3 / required | Add floor(pE) distinct nonexisting nonloop undirected edges | src/graph/perturb.py | exact counts and unchanged X/y/masks | PASS |
| G03 | (5.11),§8.3 / required | Degree-preserving double-edge swaps: Q*E ATTEMPTS, Q10, random second-edge orientation | src/graph/null_models.py | exact degree, no loops/duplicates; attempts/success/rate, >=.5 | PASS |
| G04 | §8.3 / required | Uniform simple ER graph, exact N and E, no degree preservation claim | src/graph/null_models.py | exact counts, seed and stats | PASS |
| G05 | §8.3 / required | Perturbations only change edges; preserve features/labels/masks; independent RUN_ID, configuration, hashes | graph registry | regeneration and identity checks | PASS |
| O01 | (5.15),§8.2 / required | S via fixed20000 distinct unordered nonself node pairs; same sampled pairs across models/layers | src/diagnostics/oversmoothing.py | pair hash and stable cosine, zero-vector disclosure | PASS |
| O02 | (5.16) / required | Dirichlet energy from self-loop normalized operator; equivalent degree-corrected unordered edge sum / ||H||F², scale invariant | oversmoothing.py | dense trace oracle, theoretical limit <1e-8 | PASS |
| O03 | §8.2 / required | Per-layer frozen logistic probe: train labels only, val only; common fixed method and explicit protocol | shared ExperimentRunner linear probes | no test features/targets passed to selection | PASS |
| O04 | (5.14),§9.3 / required | Untrained propagation Ahat^l X, l0..24, per-layer S/E/probe; connected-component limit caveat | propagation artifact/RUN | 10-node convergence <=1e-4; actual energy trend | PASS |
| O05 | §9.3,Lab1 (1.14),(1.16) / required | Reuse original gradient_statistics; rhoT=CEtrain/logC, GI and raw per-layer gradients; checkpoint and final-training scope explicit | graph diagnostics | direct masked CE and gradient recomputation | PASS |
| H01 | §9 / required | E1/E3 exploratory hypotheses recorded before experiments with formula basis | exploratory_hypotheses.md | real Git ancestry/time | PASS |
| E101 | §9.1 / required | E1 seed1 MLP/GCN/GAT, train/val/test/time/stats/RUN; both Gtop | E1.csv | predictions recomputed from checkpoints | PASS |
| E102 | §9.1 / required | p0,.1,.3,.5,.8,1; modeA original weights on Gp; modeB train and evaluate same Gp | deletion RUNs/evaluations | same generated graph per p across modes | PASS |
| E103 | (5.10),§9.1 / required | At p.3,.5,.8 actual isolates vs sum p^degree, relative deviation<=20%; preserve FAIL | isolation table | exact baseline degree and independent oracle | PASS |
| E104 | §9.1 boundary / required | Report p1 vsMLP, first MLP-level crossing, isolated test nodes and hE stability | E1 analysis | actual two-mode curves and counts | PASS |
| E105 | §9.1 / required | M(p) both modes +MLP line; isolates vs theory; hE vs p; transductive captions | plotting script | original saved data, visual QA | PASS |
| E201 | §9.2 / required | Git prereg BEFORE E2: original/null only, fixed X/masks/depth2/config/200epochs/seeds, expected sign/magnitude/falsifier | hypothesis.md; preregistration_evidence.json | Git blob/ancestry/timestamps | PASS |
| E202 | §9.2 / required | Three paired seeds1/2/3: GCN original, GCN corresponding Q10 null, MLP; random streams isolated | E2 runs | full paired matrix and graph/model seed metadata | PASS |
| E203 | §9.2 acceptance / required | Same degree at every node; null hE within20% hdeg; successrate>=.5; no loops/duplicates/asymmetry | E2_graphs.csv | real raw graph reconstruction | PASS |
| E204 | (5.13) / required | All3deltas,mean,ddof1 SD; nonzero same sign andabsmean>=2SD else difference unestablished | paired statistics | independent arithmetic and test predictions | PASS |
| E205 | §9.2 boundary / required | Seed1 ER exact E comparison with original/null, report degree-information limitations | ER run | actual accuracy/h0/hdeg/degree comparison | PASS |
| E301 | §9.3 / required | GCN depths[2,3,6,12,24],seed1,E1 fixed hyperparameters,no residual/norm; per-depth best val epoch | E3.csv | actual exact depth/config and checkpoint | PASS |
| E302 | §9.3 / required | lstar by val; sm from E2 originalGCN 3seeds val sampleSD; drop significant iff >2sm | E3_statistics.json | independent selection and threshold calculations | PASS |
| E303 | §9.3 acceptance / required | Classify significant drops:rho>=.9 undertrained; largeGI possible vanishing; rho<=.5 E not lower overfit; rho<=.5 E andprobe lower smoothing; otherwise mixed | E3 diagnosis | full raw evidence; largeGI convention declared beforehand, not invented PDF threshold | PASS |
| E304 | §9.3 boundary / required | Maximum-depth residual GCN, projected incompatible skip; output no ReLU; compareplain/residual; (5.14) not directly applicable | residual run | actual metrics, precise projection/skip configuration | PASS |
| E305 | §9.3 / required | Train/val vsdepth, trained/untrained logenergy, probes,S,residual comparison; explicit no significant drop if so | figures | real data + visual QA | PASS |
| MAP | §12 / required | Structure map: G,empty,p.5,null,ER,lstar,Lmax; fixed/varied factors/models/seeds/masks/RUN/direct vs auxiliary comparisons | structure_map.csv/json | complete traceability | PASS |
| C01 | §10.1 / required | MLP vs nullGCN; self-weight1/(di+1); EXTRA assigned GAT on seed1null, real comparison | CF1 run/answer | numbers and RUN_ID | PASS |
| C02 | §10.2 / required | Predict added.5E hE using hE,h0 BEFORE measurement; real graph/GCN; comparedelete.5 | CF2 prediction/run | preserved prereg prediction and actual graph stats | PASS |
| C03 | §10.3 / required | p.5 modeA vsmodeB accuracy drops and isolated test nodes; do not infer graph unused | CF3 answer | E1 actual values/RUN_ID | PASS |
| C04 | §10.4 / required | Actual high-S but near-best layerprobe witness; report S,E,probe,layer,RUN or truthful FAIL | CF4 answer | fixed descriptive witness rule, actual data | PASS |
| C05 | §10.5 / required | Only AFTER choices/models locked, posthoc test-selecteddepth vsvaldepth; no retraining or selection changes | lock manifest +CF5 | chronology and immutable selected lstar | PASS |
| T01 | §11.1 / required | MLP ignores original/empty/randomedges | tests/test_lab5_graph.py | actual pytest | PASS |
| T02 | §11.2 / required | Same weights emptyGCN=MLP <=1e-6 | tests | actual pytest | PASS |
| T03 | §11.3 / required | Edge symmetry/simple/nonloop,p endpoints; preserve X/y/masks | tests | actual pytest | PASS |
| T04 | §11.4 / required | Degree preservation and successrate on actual Cora >=.5 | tests +actual graph audit | actual pytest/data audit | PASS |
| T05 | §11.5 / required | 10-node normalized limit<=1e-4, E<1e-8, scaling invariance | tests | actual pytest | PASS |
| T06 | §11.6 / required | Mutate testlabels: same trained weights, bestepoch, hyperparameter selection | tests | actual shared-runner grid comparison | PASS |
| T07 | User engineering / required | Repro perturb/resume/RNG/RUNunique/disjointmasks/trainonlyloss/pairmath/stableS,E/depthchoice/stats/old106 regression | tests | actual JUnit and final audit | PASS |
| DEL | §13,user / required | Full configurations/logs/localweights/predictions/graphs/diagnostics/plots/RUNindexes/audit/finalGit; preserve old history | experiments/lab5; runs/lab5 | final evidence audit | PASS |
| RPT | User overrides §12 / required | No formal LaTeX/PDF/Russian report; no Lab6 | scope | filesystem/history audit | NOT_APPLICABLE |
| X01 | §14.1 / optional | SBM homophily/heterophily sweep | not requested | not run | NOT_APPLICABLE |
| X02 | §14.2 / optional | Face landmark transfer | not requested; teacher package missing | not run | NOT_APPLICABLE |
| X03 | §14.3 / optional | GAT attention importance intervention | not requested (head normalization test still required) | not run | NOT_APPLICABLE |

Model selection protocol: train every search/run for the full 200-epoch ceiling; checkpoint chosen by maximum val accuracy, earliest tie; hyperparameter ties follow the declared lr/wd grid order. All selection is locked before final test evaluation. Report node labels from the full graph only in descriptive graph statistics; those statistics never select configuration, p, Q, or models. Smoothing probes have no test evaluation.
