# Lab3 — requirements and acceptance matrix

Source: teacher `lab3_long_context_attention_1010.pdf`, revision4, 2026-10-08, all14 pages reread from original PDF. Lab2 independent audit: `experiments/lab2_verification/audit.json`; 70 regression tests passed. No Lab2 formal retraining or original record writes.

Status below records the completed implementation and real experiments; machine-readable gates, including raw interval FAIL items, are in `experiments/lab3/results.json`. PASS requires evidence, not merely code existence.

| ID | Original requirement (section/formula) | Implementation | Verification/evidence | Status |
|---|---|---|---|---|
| P01 | §7 SHA256(student_id:lab3), slices0:2/2:4/4:6 | Existing profile_gen; copy,L0=128,N=2 | experiments/lab3/profile.json; hash-only | PASS |
| P02 | §9 d128,h4,MLP256,PreNorm,AdamW,B32,no causal mask | configs/lab3.yaml; SequenceTransformer | config guards, saved resolved configs | PASS |
| D01 | §6 content vocabulary0..63; independent train/val/cal/test seeds | synthetic_sequence.py; namespace-separated SeedSequence | generator determinism/role tests | PASS |
| D02 | §6 uniform integer train L in[L0/2,L0], no longer training | Each homogeneous batch draws length; copy odd L decremented exactly as specified | batch tests; generator manifest | PASS |
| D03 | §6 copy fragment then blank half, target loss only blank | Labels -100 outside target | odd-length and loss gradient tests | PASS |
| D04 | §6 retrieval n=min(32,max(2,floor(L/8))), separate key/value markers; unique keys; query last2; random pair location, random background | Unique keys and random gap allocation between ordered four-token pair chunks | target/unique-key tests | PASS |
| D05 | §6 needle random background, unique marker, next symbol target, last query | marker index uniform0..L-3 (last token reserved query) | uniqueness/target tests | PASS |
| D06 | §6 >=200 sequences per test length, cal generated and unread untilLab8; test length never changes train | 256 examples; materialized reserved cal; read guard; stateless per-role generator | role and test-independence tests; dataset manifest | PASS |
| M01 | §8.1 four configurable representations, same training loop | Position + shared ExperimentRunner | backbone equality/config test | PASS |
| M02 | (3.5),(3.6) absolute Lmax=L0; P1 random untrained extension,P2 interpolation,P3reject | table has exactly128 rows; deterministic P1 only at inference; direct table at L<=Lmax; interpolated t=m(Lmax-1)/(L-1) only for longer lengths | identity/endpoints/rejection tests | PASS |
| M03 | (3.7) sinusoidal at arbitrary integer positions | float64 formula then float32 output | extrapolation formula test | PASS |
| M04 | (3.8),(3.9) RoPE on Q/K each head | adjacent-component rotations | shape/relative-inner-product tests | PASS |
| M05 | (3.4) noPE no mask/padding/position side-channel | equal-length batches, no padding or causal mask | permutation error<1e-5 | PASS |
| H01 | §1,§8 common Lab0 harness, provenance/seeds/logging/checkpoints | Existing runner, model/data factories, checkpoint and create_run | old70+new resume tests | PASS |
| H02 | §9 one seed specifies both initial weights and sequences | backbone constructed before optional absolute table; generator keyed by seed,role,epoch,index | paired checksums/config comparison | PASS |
| H03 | §9 E1/E2 hypotheses before experiment | experiments/lab3/exploratory_hypotheses.md | committed before real runs | PASS |
| E101 | §9 common LR once using sinusoidal val atL0; grid .0003,.001,.003 | 2000-step candidate runs, final val only, deterministic tie smallerLR | selected_lr.json; no test in selector | PASS |
| E102 | §9 start2000 steps, double all to4000/8000 if bestPE−none val<.20 | constantLR; exact-state continuation, unchanged generator protocol; 100-step complete epochs | run ledgers/checkpoints/continuation tests | PASS |
| E103 | §9.1 four modes one common seed, main absP2, no fine-tuning test | shared canonical settings, seed1 | resolved configurations/checkpoints | PASS |
| E104 | §9.1 test L0,2L0,4L0,8L0; absP1 extra2L0 | lengths128,256,512,1024; all within §6 range, no conflict | independent evaluation RUN_IDs | PASS |
| E105 | (3.13) M,ER=M/(M(L0)+1e-8),delta_pos vsabsP2 | extrapolation.py; omit near-chance noPE ER figure (raw ratio retained) | recomputable tables/plots | PASS |
| E106 | §9.1 noPE val<=1/64+.05; bestPE−none>=.20; after8000 diagnose if still fails | literal thresholds; no threshold edits | results gates; failure analysis if needed | PASS |
| E107 | §9.1 first L>L0 accuracy<=1/64+.05 or firstOOM; otherwise beyondgrid | curve first_failure + longest effective measured length | boundary.json | PASS |
| E201 | §9.2 random weights B4,>=5 lengths64,128,256,512,1024 | per-condition fresh model, fixed width/depth/device/precision/backend | profile raw records | PASS |
| E202 | §8.2 forward,training-step,peak,tokens/s;>=10 warmup; GPU sync/resetpeak | sequence_profiler.py; 30 repeats; raw times; allocator peak | profiler schema + real CUDA records | PASS |
| E203 | (3.10),(3.14) separate explicit/fused curves; global+longest2 slopes, doublingratios | explicit and forced flash if supported; samefloat16 precision, implementation logged; no silent fallback | E2.csv/slopes/plots | PASS |
| E204 | §9.2 explicit1.6..2.4 when doubling>=3.03; implicitmemory .8..1.2; otherwise decompose L²d andLd², accepted analysis | separate quantitative slope and explanatory gates | decomposition.json, actualpeak distinct from tensor/theory | PASS |
| E205 | §9.2 OOM record and supplement intermediate lengths to>=5 successes | safe_measure records exception/recovery; allowed intermediate lengths ifneeded | OOM/recovery test; run ledger | PASS |
| E301 | §9.3 choose A/B fromE1, onlyPE differs; prereg L*>L0,sign,falsification,fixed vars includingP2/backend | immutable hypothesis.md committed before new seeds | preregistration_evidence.json and git ancestry/time | PASS |
| E302 | §9.3 common seeds1,2,3; exactmatchingE1seed1 reusable | 2arms ×3, reuse matched seed1 | pairing/sourceRUN_IDs, configs/checksums | PASS |
| E303 | (3.15) paired3 deltas,mean,ddof1SD, nonzero same sign andabsmean>=2SD | extrapolation.paired; else различие не установлено | E3.json and unit tests | PASS |
| A01 | §8.3 each block/head H,D,mass radius16,max,answerentropy; targetmass onretrieval/needle; averageexamples notheads | attention_stats.py; all256 examples atL0/2L0 one trainedmodel | raw tensors + per-head/query JSON + heatmaps | PASS |
| A02 | §8.3 interpret concentration/length/targetmass, no causalclaim | structured interpretation linked toRUN_IDs; copy targetmass supplemental | attention_analysis.json | PASS |
| C01 | §10Q1 sinusoid ER atmaximumL and finite generalization | saved E1 data | counterfactuals.json Q1 | PASS |
| C02 | §10Q2 quadruple L ->16× attentionelements versus realpeakratio/slopes | actual E2pairs | counterfactuals.json Q2 | PASS |
| C03 | §10Q3 absP1/P2 measured2L0 comparison | extra independent evaluation | counterfactuals.json Q3 | PASS |
| C04 | §10Q4 RoPE vsabs atL* doesnotprove lengthinsensitivity, fullcurves+3deltas | E1/E3 evidence regardless observed leader | counterfactuals.json Q4 | PASS |
| C05 | §10Q5 doubledwidth fixedL new forwardmeasure, compare2×/4× | separate profile RUN_ID d256 | counterfactuals.json Q5 | PASS |
| T01 | §11.1 allowed rowsum tolerance1e-5, maskedweights0 | test_attention_rows_and_mask | pytest | PASS |
| T02 | §11.2 noPE equivariant maxerror<1e-5 | test_no_position_permutation_equivariance | pytest | PASS |
| T03 | §11.3 sinusoid beyondL0 | test_sinusoid_extrapolation | pytest | PASS |
| T04 | §11.4 RoPE shape/relative innerproduct tolerance1e-5 | test_rope_shape_and_relative_identity | pytest | PASS |
| T05 | §11.5 P2identity atLmax tolerance1e-6 | test_absolute_rules | pytest | PASS |
| T06 | §11.6 deterministicdata/uniquequerykey/uniquemarker/stabletrain | generator tests | pytest | PASS |
| T07 | User engineering:resume,RUNunique,roles,LRvalonly,statistics,profiler,attentionstable,OOMrecovery,oldregressions | tests/test_lab3_sequence.py and oldsuite | actual JUnit records | PASS |
| O01 | §12–13 preserved configs/RUN_IDs/logs/checkpoints/plots/statistics/gates | experiments/lab3 +runs/lab3; rawweightslocal ignoredbyGit | final audit +commit | PASS |
| O02 | §12 report requirement explicitly deferred by current user | No report/LaTeX/PDF creation; data ready for later writing | gitdiff | NOT_APPLICABLE |

## Optional tasks (§14), excluded from mandatory completion

| ID | Requirement | Decision | Status |
|---|---|---|---|
| X01 | Train localattention windoww4/16/64+full, reachabilityR<=Nw | Not requested | NOT_APPLICABLE |
| X02 | Four dependencydistances M(d), optionallocal reachability | Not requested | NOT_APPLICABLE |
| X03 | Attention intervention distantweightzero/uniformhead andaccuracy | Not requested; no causal claims | NOT_APPLICABLE |

§15 conceptual questions support interpretation; all five §10 counterfactual questions are mandatory. No Lab4 execution.

## Fixed decisions before results

AdamW weight_decay=.01, fixedLR (no schedule; PDF leaves these unspecified), float32 training, explicit attention, GELU, no dropout, 256 validation and256 test sequences perlength. 100 batches per complete dataepoch; fresh deterministic sequences per epoch. Each batch samples integer requestedL uniformly; Copy reducesoddL byone as PDF requires. This induces paired even effective lengths and is disclosed rather than silently sampling onlyevens. Validation andtest use independently keyed seeds and exactrequestedlength. E1/E3 backboneinitialization is identical for same seed across modes because optionalabsolute table initializes last. Formal evaluations always use finalweights, never pick checkpoints bytest. Cal materialized atL0 andnever opened. LRties select smallestLR. No reports modified.

## Final evidence

Final88 tests pass (old70 +18sequence tests). All required experiment/protocol checks PASS. E2 raw memory interval checks FAIL for both implementations and are preserved in E2_analysis.json/results.json; PDF§9.2 alternative decomposition is completed separately. Flash itself is unavailable in this Windowsbuild; forced supported memory-efficient attention supplies the second required curve. Two exploratory failed attempts (deterministic NLL2d smoke and unavailable Flash) are retained. No mandatory blocker, no report generated, no Lab4 begun. Six scientific figures visually inspected. See experiments/lab3/audit.json and checkpoint_index.json.
