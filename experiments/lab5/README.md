# Lab5 reproducibility and evidence

This folder contains experiments and checks, not a formal Russian/LaTeX/PDF report. Lab6 is out of scope.

Use the existing `.conda-env/python.exe` from the repository root. PyTorch2.7.1+cu126 and torchvision0.22.1+cu126 were preserved. Only PyG2.7.0 and its missing pure-Python dependencies were added; every preexisting package version was constrained and unchanged. `docs/environment/lab5-requirements.lock.txt` is the new environment lock; the historical root lock was not overwritten. `budget.json` records real CPU/CUDA measurements, and all formal runs use CPU2threads, deterministic float32.

## Evidence navigation

- `EXPERIMENT_SUMMARY.md`: Chinese outcome summary, including quantitative failures.
- `../../docs/requirements/requirements_lab5.md`: complete mandatory/optional requirement matrix.
- `profile.json`, `teacher_source.json`, `data_audit.json`, `split_decision.json`: actual profile, original PDF hash, graph/raw-file hashes and exact public-mask indices. Teacher split independently UNVERIFIED; user explicitly authorized public-mask substitution.
- `hypothesis.md`, `preregistration_evidence.json`: immutable E2 hypothesis and real Git commit before every E2 RUN. `exploratory_hypotheses.md` preceded E1/E3. `CF2_prediction.json` preceded the added-edge graph and its GCN.
- `E1_selection_lock.json`: all12 candidates and val-only selected lr/wd/epoch. `all_training_selection_lock.json`: all graph-model training complete and val-depth fixed before CF5 test argmax.
- `E1.json`, `E2.json`, `E3.json`, their CSV tables, `E3_all_layers.csv`: measured outcomes and explicit RUN_IDs. `run_index.json` links training, evaluation and smoothing RUNs.
- `counterfactual_answers.md`: all five measured counterfactuals. `structure_map.csv/json`: fixed/varied factors and comparison limits.
- `graphs/*.json`: original-edge perturbation parameters, random seeds, generation RUN_ID, immutable node-data fingerprints, actual graph statistics/hashes. Degree-preserving null records Q*E attempts, successful swaps and acceptance checks.
- `figures/*.svg` and `*.png`: vector exports and previews from saved results. `figure_QA.json`: visual inspection record.
- `final_audit.json`, `checkpoint_inventory.json`, `evaluation_recalculation.json`, `graph_regeneration_audit.json`: independent results recalculation and integrity audit.
- `../lab4_verification`: independently recomputed old results, 334 loadable checkpoints and3925 historical protected-file hashes; old Lab4 Q2 stays BLOCKED.

## Existing-run verification (no graph-model retraining)

```powershell
.\.conda-env\python.exe -X utf8 -m pytest -q --junitxml=docs/environment/tests-lab5-final.xml
.\.conda-env\python.exe -X utf8 -m reports.tools.lab5_artifacts
.\.conda-env\python.exe -X utf8 -m reports.tools.lab5_audit
.\.conda-env\python.exe -X utf8 -m reports.tools.lab5_finalize
```

Artifacts and audit regenerate only Lab5 analyses; protected Lab0–4 records are not rewritten. `reports.tools.verify_lab4_before_lab5` makes a separate Lab4 verification copy and does not train Lab4.

## Training and recovery

Shared runner entry point (already registered graph models/data; not a separate training framework):

```powershell
.\.conda-env\python.exe -X utf8 -m src.train --config configs/lab5.yaml --set experiment.purpose=formal
```

`--set model.kind=mlp` or `gat` chooses the model; graph_path must agree in data/model. Every invocation creates a new RUN_ID and full config/manifest. `--resume <RUN>/checkpoints/last.pt` restores optimizer/scheduler/scaler/Python/NumPy/Torch/CUDA/loader RNG, with strict source/config/data/split checks. A resume always creates a new RUN_ID; never bypass the source-hash check after code changes. `experiments/lab5/PAUSE` is checked at complete epoch boundaries. Formal models use200epochs; smoke/budget records are separately identified and excluded.

The serial controller supports `e1`, `e2`, `e3_train`, `analysis` via `-m reports.tools.lab5_execute <stage>` and consults its persistent index before doing work. E2 refuses to run without a real Git hypothesis blob and ancestor check. `analysis` trains frozen linear probes if their stage is missing, so it is not listed above as a read-only verification command. Do not delete indices or overwrite records to rerun an experiment. Fresh reproductions should use an isolated clean checkout/output namespace and preserve prospective preregistration chronology. Exact historical runs use the commit and source hash in their manifest, not an assumed current HEAD.

Local large artifacts are deliberately ignored by Git: `data/planetoid/Cora/raw`, `data/lab5/*.pt`, `data/lab5/features`, and `runs/lab5/*/checkpoints`. Their paths and SHA256 are recorded; no fabricated replacement is supplied if missing. Raw Planetoid files are downloaded by PyG from its declared original source. The prepare utility may be used in a fresh checkout; it should not overwrite the original data/protocol files of this completed experiment.

## Interpretation restrictions

All labels may enter descriptive hE/h0/hdeg, never model selection. Only train targets enter optimization/probe standardization; val selects configuration/checkpoint/depth; test is descriptive, with CF5 calculated only after the model/val lock. See the disclosed E1 seed1/depth2 prior exposure. rhoT/GI selected-checkpoint diagnostics and full-training median gradients are both retained. GI>=6 is a disclosed analytical convention, not a teacher-specified threshold. Constant training columns in probes use unit scale; this was fixed before any formal probe outcome was observed. Three seeds quantify uncertainty only on this single split. Graph topology benefit is established by original-versus-degree-preserving-null differences, not Gtop alone. Poor deep-network accuracy is not automatically oversmoothing.
