# Lab 2 conditional preregistration — before formal LR search and E1

The canonical CIFAR ResNet18 and ViT (P4, six blocks, D384, six heads,
MLP1536) are compared without pretraining. The primary comparison is
parameter matched; MAC differences will be reported, not hidden.

All primary runs use B128, AdamW weight decay .05, 500-step linear warmup
and cosine decay to S10000 actual optimizer updates, A1, float32. Each
architecture selects LR independently from .0003, .001, .003 using final
validation accuracy at r1, with ties resolved by lower validation loss,
then lower LR. Those r1 seed1 runs are reused by E1. All fractions use the
same balanced nested class permutations, never resampled across models/seeds.

E3 regime selection is conditional: choose the profile fraction with largest
absolute final validation accuracy difference (ViT minus CNN), ties choose
smaller r. Freeze that selection before accessing test results. Predict
the same sign on test as on validation. The finalized numerical prediction,
selected fraction/LRs and original seed1 RUN_IDs will be committed before
new E3 training seeds2,3. Seed1 is explicitly the reused exploratory E1 run,
as required by the PDF; it is not represented as an independent replication.

The hypothesis fails if the test difference has the opposite sign or if
the paired three-seed differences do not have one common nonzero sign and
absolute mean at least twice sample standard deviation (ddof1). An
inconclusive or reversed result is retained. No test result changes LR,
fraction, architecture, S, seed list, or the selection rule.

E2 uses the full-fraction E1 checkpoints with blur, contrast and patch
shuffle at levels0–3. The map evaluates all five fractions at levels1–3
plus clean. All paired inputs have identical per-image corruption seeds.
CF2 adds only the small-fraction CNN equal-epoch S506 intervention; CF4
profiles P2 without training; CF5 evaluates zero-filled translations.
Optional augmentation/transfer/CKA training is outside this mandatory suite.

Smoke and 336-update resource budgets are development evidence only and
are excluded from formal scientific results. No report is produced at
this stage, in accordance with the user's instruction.
