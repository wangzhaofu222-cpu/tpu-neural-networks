# Lab2 mandatory experiment protocol (PDF revision 4, all 14 pages reread)

No LaTeX/PDF report is produced in this stage. Existing Lab0/1 artifacts stay immutable. Course data protocol is unavailable; reuse saved Lab0 split, without claiming teacher-index identity.

## Models, data and fixed training

- GPU canonical pair: torchvision ResNet18 with 3x3/stride1 stem and no maxpool; torchvision VisionTransformer image32/P4/L6/D384/H6/MLP1536, class token, pre-LN, GELU. No pretrained weights. Compare **param_matched** (<=20% parameter difference); report MAC mismatch rather than claiming equal computation. CPU reduced variant is not substituted while GPU is available.
- CIFAR train43000 / val5000 / cal2000 / test10000. Never use cal. Fixed base-seed nested per-class subsets: floor(4300*r) each class. Persist actual original indices and their fingerprint in manifest.
- Profile P3adf02: fractions .05,.15,.30,.60,1; corruptions blur,contrast,patch_shuffle; A2 randaugment+cutmix. Verify through existing lab_pick, keep original ID out of files.
- Every primary training has S=10000 actual optimizer updates, B128, AdamW weight decay .05, warmup5% updates then cosine, A1 crop32/padding4+horizontal flip. No early stopping. Float32, no AMP. Separate LR grids .0003,.001,.003 at r1, seed1, all full S; choose highest **final val** accuracy, tie lower val loss then smaller LR. No other tuning.
- Intermediate validation is omitted uniformly; all training loss/accuracy curves are logged, final val selects LR/mode. Last-S checkpoint supplies final evaluation (no best-epoch selection). Save recoverable full-data-epoch checkpoint boundaries at approximately1000-update intervals; final partial data epoch is explicitly marked and used for inference. Existing epoch-boundary resume guarantee retained.
- E1/E2 exploratory seed is literal1. E3 common literal seeds1,2,3. Profile subset selection uses base_seed, independent of training seed.

## Work and acceptance matrix

| Item | Mandatory coverage and numerical gates |
|---|---|
| Model API | logits(B,10), forward_features(B,d), optional intermediate features; classes/position embeddings and P recorded |
| Resources | total/trainable params, per-image MAC including all MHA QKV/output projections and QK/AV, analytic-vs-hook error<=1%; memory; B1 latency, B128 throughput, median30 samples after10 warmups, CUDA synchronization; training time |
| Data/augmentation | nested balanced fixed indices; A0 deterministic; A1; registered profile A2 with label-correct CutMix; implement alternative MixUp policy for completeness |
| Corruptions | implement all6, s0 exact identity; raw[0,1] before normalization, clip; gaussian std .04/.08/.12; blur sigma .5/1/1.5 kernel2ceil(3sigma)+1; contrast .6/.4/.2 around mean brightness; occlusion grey square6/10/14; translation zero-fill2/4/8 random seeded axis/sign; patch grid upper-left, block16/8/4 seeded permutation |
| E1 | 2 models x5 fractions =10 conditions, val and test accuracy, fixed10000 updates, actual/equivalent epochs, loss curves, delta=ViT-CNN, r*=first profile fraction with delta>=0 or explicit none, runtime-vs-accuracy r1 |
| E2 | reuse E1 r1 checkpoints; profile3 corruptions xlevels0..3 x2models, clean equality exact; relative drops epsilon1e-12, per-corruption mean overlevels1..3; leader reversal only opposite signs with BOTH absolute gaps>=.01 |
| E3 | selected mode by val only; pairedtest differences3, sample std(ddof1), same nonzero sign and abs(mean)>=2std or difference not established; preserve all RUN_IDs |
| Map | fractions x(clean + profile corruptions/levels) -> delta; evaluate each E1 checkpoint on identical seeded transformed images; explicitly fixed factors |
| CF1 | own smallest/full delta, no extrapolation beyondr1; mechanisms besides sample count |
| CF2 | additional CNN at smallest fraction with S_cf=ceil((10000/ceil(43000/128))*ceil(2150/128))=506; compare its test accuracy with full-S E1; derived shorter warmup/cosine horizon disclosed |
| CF3 | resource and training-time ratios, param/compute matching status |
| CF4 | fresh P2 ViT profiling only: N'=257 vs65, analytic+measured MAC and attention quadratic term ratio; parameter change |
| CF5 | additional translation evaluations2/4/8 for BOTH E1 r1 models (translation absent from profile); compare patch4/overallCNNstride8 and augmentation support |
| Tests | all6 teacher categories: patch count+roundtrip; shapes+profiler immutability; exact convMAC; nested balanced subsets and exactS at all fractions; A0 deterministic; all6 corruptions preserve labels,s0 identity,seed reproducibility. Retain Lab0/1 regression; add step-scheduler/checkpoint tests |

## Preregistration chronology and test isolation

PDF explicitly reuses E1 seed1 in E3 while selecting its regime after E1/E2. To avoid retrospective claims: commit an **initial conditional E3 selection rule before E1**, then lock the selected fraction using val before any test results, and commit a finalized Lab2-only hypothesis before NEW E3 seeds2,3. Clearly label seed1 exploratory and reused, exactly as §9.3 requires. Never overwrite Lab1 root hypothesis.md.

Initial selection rule: argmax absolute final-val ViT−CNN difference over profile fractions, ties smaller fraction; expected test sign equals selected val sign. Final hypothesis records numeric expected difference, LR and original seed1 RUN_IDs, and falsification rule. E2 test observations do not alter the val-locked mode or LR.

Training count: 6 full-S LR candidates (selected r1 pair reused in E1), 8 additional E1 runs, 4 NEW E3 runs (seed1 reused), 1 CF2 short-budget run = **19 unique formal training executions**, 180506 updates total. No mandatory A0/A2 training: augmentation intervention, face transfer, and representation/CKA series in §14 are OPTIONAL; face pack absent. A2 is nevertheless implemented and tested.

Budget: first smoke then complete real r1 epoch per canonical model; record observed seconds/update and estimate all180506 updates plus profiling/evaluation. Do not reduce fractions, seeds, architecture or S to fit convenience. Use measured budget to configure time limits; raw failed/interrupted attempts remain separate.

All evaluation rows save model-checkpoint hash, input transformation fingerprint, source RUN_ID and independent evaluation RUN_ID. Final results/acceptance/counterfactuals/learning curves/map/checkpoint index derive automatically from raw records. Distinguish execution completion, numeric PASS/FAIL, and genuine BLOCKED. Stop after Lab2.
