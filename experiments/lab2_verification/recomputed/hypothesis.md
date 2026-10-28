# Lab 2 E3 finalized hypothesis

The conditional selection rule was committed in
a49fccf3907e64b00e1cf83b05ce755fcf126e80 before formal E1/LR search.
The original text is preserved as hypothesis.initial.md and in Git history.
E1 seed1 is explicitly exploratory and reused, as required by PDF9.3;
only seeds2,3 are new confirmatory training. No test measurement chooses
this regime: selection.json is a timestamped validation-only lock.

Comparison: canonical CIFAR ResNet18 versus ViT P4/L6/D384/H6/MLP1536,
parameter-matched mode, fixed fraction r=0.05. Fixed balanced subset
indices from base_seed, common training seeds1,2,3, S10000 actual updates,
B128, A1, AdamW wd.05, warmup500 then cosine, float32, no pretraining.
CNN LR=0.003, ViT LR=0.0003, independently selected
once on r1 final validation from the preregistered three-point grid.

Expected mean test difference ViT minus CNN: approximately -0.256800
(-25.680 percentage points), with negative sign,
based on the selected seed1 final-validation gap, equation2.8. This is
a prediction, not an observed test mean. Locality and shared spatial
weights in equations2.1–2.2 versus learned global attention and positional
embeddings in2.3–2.5 motivate possible differences at fixed finite data/S.
No causal attribution to a single mechanism is claimed.

Refutation: an opposite/nonmatching mean sign, mixed or zero paired signs,
or abs(mean_delta)<2*sample_std_delta (ddof1) rejects the stated stable
architecture difference. All three paired differences must be shown;
uncertainty is reported even if it contradicts this expectation.

Seed1 source runs:
CNN 8705d516f9c58ae4458c-2297f86c299c4b998e5a33258a6b8326
ViT 76a6bee7ce9731bfa4fa-8edbe989a96e427c82f75e87ddd2c0ab

The subset is held fixed across seeds. Three seeds describe initialization
and batch ordering variability; they do not cover new subset draws,
alternative recipes, or natural distribution shifts. E2 test observations
cannot alter the validation-locked fraction, LR, seed list or stopping.
