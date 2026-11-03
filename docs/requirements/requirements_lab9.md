# Lab9 最终需求检查表

教师原PDF14页完整重读；数据和实验均来自真实归档。报告依用户指令暂不制作。

| ID | PDF原始要求 | 实施方法 | 验证方法 | 状态 |
|---|---|---|---|---|
| H01 | §1/§13: reuse Lab0 EXP_ID/RUN_ID harness | Shared ExperimentRunner; additive Lab9 macro-F1/early-stop only | regression tests and manifest trace | PASS |
| D01 | §6: genuine RAVDESS1440 clips/24actors/8classes | Official24speech archives, officialMD5 and clip IDs | complete acquisition manifest | PASS |
| D02 | §6: actor-disjoint14/3/2/5 inherited split | Deterministic backup1729; teacher split not found | actor lists/hash/counts | UNVERIFIED |
| D03 | §6: cal not read; val selection; test final | No cal media extraction; explicit role guards | filesystem and sample IDs | PASS |
| D04 | §6: train-only normalization | Audio mel and cached clip features train only; visual upstream train normalization retained | recalculate train statistics | PASS |
| P01 | §7: four SHA256 lab_pick fields | Private existing ID; preserve contradictory source/architecture fields | independent hash slice recomputation | PASS |
| P02 | §7: effective visual encoder identity | User-authorized visual-source precedence, CNN versus generic vit_small conflict retained | strict original checkpoint/hash/config | UNVERIFIED |
| T11 | §8.1.1:5FPS frozen preclassifier visual features | 224face inputs; skip face_found=false; cache byclip | weights/hash/eval/padding checks | PASS |
| T12 | §8.1.2/(9.3):logmel64,400,160,centerTrue | Explicit HTK power filters and periodic Hann; 56000->64x351 | actual shape test | PASS |
| T13 | §8.1.2:AudioCNN train/val selection then freeze | Shared runner; val macro-F1 best; fixed budget/early stopping | checkpoint curves, no test access | PASS |
| T14 | §8.1.3/(9.10): unified explicit mask | model(visual,audio,mask,align); zero token inside model | invariance/illegal mask tests | PASS |
| T15 | §8.1.4/(9.4): integer half-open synchronization | Integer samples; offsets/shifts/r; explicit valid coverage | boundary/no-gap/no-overlap tests | PASS |
| T16 | §8.1.5:passports include time and encoder hashes | Resolved config + feature metadata + cache identity | raw cache/checkpoint provenance | PASS |
| T21 | §8.2.1:video/audio/early/gated/late registry | One hidden nonlinear head; scalar profile gate | normalization/limit tests | PASS |
| T22 | §8.2.2:separate fusion/head/frozen parameter counts | Explicit parameter breakdown and external frozen source | independent parameter sum | PASS |
| T23 | §8.2.3:matched controls<=1% or nearest width | Exact integer width optimizer, record step and nearest error | enumerate adjacent widths | PASS |
| T24 | §8.2.4:late0.5 and val-selected grid0:.1:1 | Use cached unimodal probabilities; val macro-F1 only | lambda provenance and limits | PASS |
| T25 | §8.2.5:per-clip gates and conditional summaries | Mean, sample SD, emotion, correct/error | per-clip recomputation | PASS |
| T31 | §8.3.1/(9.12):input noise, time mask, blur/dropframes | Waveform measured power; reuse Lab8 blur parameters; deterministic | SNR<=.1dB,level0 identity | PASS |
| T32 | §8.3.2:reencode corrupt raw signal/frames | Same frozen encoders; separate corrupt cache identities | input/cache hashes | PASS |
| T33 | §8.3.3/(9.9):same-model contribution | Save full/withoutaudio/withoutvideo/shuffle probability vectors | independent macro-F1 and L1 | PASS |
| T34 | §8.3.4:distinguish separately-trained unimodal probs | Explicit source RUN_ID per predicted array | source registry audit | PASS |
| T35 | §8.3.5:ProtocolB undefined metrics reject | ValueError accuracy/macroF1/ECE/Risk, no fabricated label | mandatory test6 | PASS |
| E11 | §9.1:single-seed exploration all strategies | Controlled budgets, sources, classes, bestval checkpoints | val/test metrics and confusion | PASS |
| E12 | §9.1:best strategy/controls chosen byval | Lock validation ranking before main test comparison | selection evidence | PASS |
| E13 | §9.1/(9.8):G,Gint,Gctrl | Per-strategy test gain with matching-control scope | recompute from raw predictions | PASS |
| E14 | §9.1:hypothesis.md realGit before main | F/U/fixedconditions/direction/falsifier/seeds1,2,3 | Git blob and timestamp checks | PASS |
| E15 | §9.1/(9.14):three actual paired seeds | Train F/U after prereg; shared frozen features | three differences, mean, ddof1SD | PASS |
| E16 | §9.1:all same sign and absmean>=2SD | Preserve null/reverse result and prediction FAIL | independent teacher-rule calculation | PASS |
| E17 | §9.1:parameter count and confusion/perclass figures | Automatically plot saved outputs | byte regeneration and visual QA | PASS |
| E21 | §9.2:modality dropout profile probability | pdrop=.5; full=.5/single=.25each; neverbothmissing | sampling test and training manifest | PASS |
| E22 | §9.2:all fusion Full/NoAudio/NoVideo | Same model zero null token; actual majority M0 | R,Rtilde,delta and undefined reasons | PASS |
| E23 | §9.2:audio mask versus no-fixed-point shuffle | Fixed derangement; video/labels unchanged | L1 and changed prediction counts | PASS |
| E24 | §9.2:compare Rv-Ra versus MV-MA | Actual independent models versus same-model interventions | saved probability and source separation | PASS |
| E25 | §9.2 boundary:SNR30,20,10,0,-5 both gated models | Same300testclips, waveform corruption then reencoding | perclip SNR/F1/gate and no-audio reference | PASS |
| E26 | §9.2:highest gridSNR worse than noaudio or absence | Do not presuppose gate shift | grid scan and explanation | PASS |
| E31 | §9.3:>=200 sameemotion and>=200 differentemotion pairs | 300unique ordered pairs each, differentactors, fixedseed | pair identity/label tests | PASS |
| E32 | §9.3:early/gated/late0.5 evaluate A/B | A F1/accuracy/conf; B rv,ra,r0,deltaij/conf and true gates only | raw pair prediction files | PASS |
| E33 | §9.3:rv+ra+r0=1; B undefined | Raise on invalid metric requests; gates N/A when absent | independent counts | PASS |
| E34 | §9.3:detect inconsistency AUROC1-conf anddeltaa | Anegative/Bpositive; deltaa differs fromdeltaij | independent AUROC and baseline.5 | PASS |
| E35 | §9.3:confidence/gate figures and actor confound | Original/A/B histograms; identity versus emotion limits | real histograms and discussion | PASS |
| Q1 | §10.1:weak modality use not forced | Own Rtilde and mean delta evidence | RUN_ID-backed measured answer | PASS |
| Q2 | §10.2:double control hidden width and retrain | New genuine training run and revisedGctrl | checkpoint/curve/probabilities | PASS |
| Q3 | §10.3:audio shuffle versusmask | Use measured output changes andRa | RUN_ID-backed measured answer | PASS |
| Q4 | §10.4:Lab8 level1blur real reencoding | MeasureM_AV,gate,same-modelvideo intervention | RUN_ID and corrupt input cache | PASS |
| Q5 | §10.5:gate versusSNR not guaranteed | Use actual E2 curves; no invented trend | RUN_ID-backed measured answer | PASS |
| A01 | §11:all six mandatory test groups | Unit tests + genuine data/experiment audit separated | JUnit and independent audit | PASS |
| A02 | §12:all results traceable toRUN_ID and automatic plots | Raw probability arrays, tables, deterministic generation | independent regeneration | PASS |
| A03 | §13:all mandatory experiments and prereg complete | Final checklist distinguishes experiment/prediction/protocol | audit and clean Git commit | PASS |
| N01 | §12:formal report | Explicitly deferred by user | No LaTeX/PDF changes | NOT_APPLICABLE |
| N02 | §14:crossattention/calibration/time-shift study/regression | Additional tasks not requested; synchronization shift still unit-tested | scope statement | NOT_APPLICABLE |
