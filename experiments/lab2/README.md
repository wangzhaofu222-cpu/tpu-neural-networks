# Lab 2 evidence and execution

This directory contains data-derived outputs, not a LaTeX/PDF report.
Lab0/1 historical artifacts are not regenerated.

Shared training harness entry points (from repository root):

```powershell
.conda-env/python.exe -X utf8 -m scripts.lab2_suite smoke
.conda-env/python.exe -X utf8 -m scripts.lab2_suite budget
# Commit clean source and conditional hypothesis before formal LR/E1.
.conda-env/python.exe -X utf8 -m scripts.lab2_suite lr
.conda-env/python.exe -X utf8 -m scripts.lab2_suite e1
.conda-env/python.exe -X utf8 -m scripts.lab2_evaluate lock
# Write finalized numerical hypothesis from validation; commit it, then seal.
.conda-env/python.exe -X utf8 -m reports.tools.lab2_operations finalize-hypothesis
.conda-env/python.exe -X utf8 -m reports.tools.lab2_operations seal
.conda-env/python.exe -X utf8 -m scripts.lab2_evaluate e1
.conda-env/python.exe -X utf8 -m scripts.lab2_suite e3
.conda-env/python.exe -X utf8 -m scripts.lab2_suite cf2
.conda-env/python.exe -X utf8 -m scripts.lab2_suite profile2
.conda-env/python.exe -X utf8 -m scripts.lab2_evaluate e3
.conda-env/python.exe -X utf8 -m scripts.lab2_evaluate cf2
.conda-env/python.exe -X utf8 -m reports.tools.lab2_resources --arm cnn
.conda-env/python.exe -X utf8 -m reports.tools.lab2_resources --arm vit
# Also execute the all-corruptions command below for each r1 checkpoint.
.conda-env/python.exe -X utf8 -m reports.tools.lab2_results
```

Every ledger alias explicitly reuses a checkpoint, with zero new training
time. Unique formal training count is19 (six LR, eight other E1, four new
E3, one short CF2), not the number of logical experiment labels. Smoke,
budget and profiling/evaluation records are excluded from that count.

`selected_lr.json` uses final validation only; `selection.json` locks E3
fraction before any test access. `hypothesis.initial.md` preserves the
conditional rule; finalized `hypothesis.md` and Git evidence precede new
E3 seeds2,3. Reproduction requires a fresh output directory/new hypotheses,
not overwriting old seed1 evidence with retrospective predictions.

`E1.csv`, `E2.csv`, `E2_means.json`, `E3.csv`, `trainability_map.csv`,
`CF5_translations.csv`, `counterfactuals.json`, `results.json` and `figures/`
are generated exclusively from saved run data. Each evaluation manifest
records parent run, checkpoint SHA256 and actual normalized-input SHA256.
Corruption values and seed mapping are in `corruption_protocol.json`.

Single-model evaluation of all six corruptions and levels0–3:

```powershell
.conda-env/python.exe -X utf8 -m reports.tools.lab2_operations all-corruptions --source runs/lab2/<RUN_ID>
```

Resource passports use the common profiling command with `configs/lab2.yaml`
and optional `--set model.name=cifar_vit`. Mandatory time samples are 30
after10 warmups, synchronized CUDA. Peak memory distinguishes training and
eval-forward measurements. MAC convention is one multiply-accumulate,
including QKV/output projections and attention products; norm/softmax/
GELU/bias adds are excluded explicitly.

Primary resource passports are additionally measured in separate fresh
processes (`Resource_cnn_isolated`, `Resource_vit_isolated`), without
optimizer state. Training time/peak are taken from the actual r1 runs;
the original after-training profiler records are retained as well.

`checkpoint_index.json` lists local last/best checkpoints and hashes.
Weights are retained locally under each run; Git excludes binary weights.
Use the recorded resolved configuration to rebuild the model and load
`checkpoints/last.pt`; both architectures expose `forward_features` for
later Labs8/9. A final partial-data-epoch checkpoint is valid for inference,
but only complete-data-epoch checkpoints support exact training resume.
No Lab3 work is initiated.

External interruption during CNN r.6: original logs/checkpoint are retained,
`interruption.json` records the event. Recovery from complete epoch5/update1010
creates a child run with full optimizer/scheduler/RNG state;499 already logged
updates must be replayed. Thus19 completed training conditions correspond to20
formal process invocations, including one interrupted parent. Logical scientific
updates total180506; recorded physical updates including replay total181005.
Exact monotonic total time of the externally terminated parent is unavailable;
its recorded complete-epoch time is reported separately as a lower bound.
