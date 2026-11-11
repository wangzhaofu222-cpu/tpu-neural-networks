# 2026-11-06 revision: fixed protocol before new results

User decision: design the missing colour control; adopt the existing Lab5/9 choices.
Historical results and hypotheses remain immutable. New work has separate RUN_IDs.

## Lab1 replication
Audit the original residual formulas and first-block c definition. Repeat the full
E2 3 architectures x 4 depths, 3 epochs each, using the original seed, LR .003,
data, widths and thresholds. Do not search seeds or change thresholds. A numerical
failure is recorded at its actual stopping point. Compare original and repeated
initial moments and trainability. If the original failure reproduces without a
code error, retain it and explain the finite-width cross term, rather than repair
the measured result. E3 is unchanged; no retrospective hypothesis edit.

## Lab4 Q2: HueCIFAR-10 control v1
Reuse the original CIFAR train/val/test image identities (43000/5000/10000), never
cal. Discard semantic labels. Within each role independently, assign exactly
balanced random hue classes 0..9, with fixed NumPy PCG64 seeds 4601/4602/4603.
Hue is (class+.5)/10 plus uniform jitter [-.015,.015]; saturation .8; value is
.25+.65*(.299R+.587G+.114B), RGB in [0,1]. Convert HSV to RGB, round to uint8.
This preserves spatial luminance texture while making hue the sole class rule.
Different roles use different original images; hue labels do not depend on image
content. No stochastic augmentation is used for frozen feature extraction.
Use the original train-only CIFAR normalization expected by the encoders.

Freeze the two already-trained E2 25-epoch encoders: full T and T without
ColorJitter. Extract h (512D). Train linear probes with the shared ExperimentRunner,
200 full-batch epochs, SGD lr .1, momentum .9, cosine schedule, L2 grid
[0,.001,.01], seed 1 and identical balanced examples. Standardize features using
only train. Select L2 by final val accuracy, ties smaller L2; evaluate test once.
Prediction: removing colour jitter preserves more hue information, so the probe
without jitter is expected to perform better. Any null or reversed result is
retained; one seed is descriptive, not a three-seed established comparison.
Save generated labels/indices, input hashes, frozen checkpoint hashes, feature
caches, all six probe checkpoints, probabilities and confusion matrices.

## Adopted Lab5/9 configurations
Lab5: PyG Planetoid Cora public masks.
Lab9: deterministic actor split seed1729, 14/3/2/5 actors for train/val/cal/test;
visual encoder is the explicitly sourced Lab2 CNN checkpoint (source field takes
precedence over the conflicting generic architecture field). Existing runs stand.
These are the current project protocols under the user's explicit decision.
