# Lab9 reproducibility and artifact map

The formal evidence is archived. Do not rerun the orchestration stages over this archive: they now reject overwriting completed exploration/main results. No Lab10 is invoked.

Use the existing `.conda-env/python.exe` and `requirements-lab9.lock.txt`, from the repository root. All raw data, cached tensors, predictions and trusted checkpoints remain on the local D drive. Binary `.pt`/data files follow the project's existing Git-ignore policy; their identities and readable metrics/configs are committed.

## Verify without retraining

`python -m pytest -q` runs all Lab0–9 tests. `python -m reports.tools.lab9_audit` checks preserved Lab0–8 files, official archives, actor isolation, train-only normalization, canonical alignment/validity, caches, checkpoints, paired statistics, raw probability metrics and regenerated figures. The audit saves an explicitly **post-experiment** validation-probability replay and verifies the original locked val scores/lambda grid; this replay is not represented as a preregistered artifact.

`python -m reports.tools.lab9_plots tmp/lab9_plot_replay` rebuilds figures/tables in a separate destination. Compare with `experiments/lab9/figures/artifact_hashes.json`.

## Reproduce an individual training run

Load its `runs/lab9/<RUN_ID>/config.resolved.yaml` and call `ExperimentRunner(config).run()`. The shared Lab0 runner creates a **new RUN_ID**, leaving original checkpoints and summaries untouched. Use the Git commit from that run's manifest for exact code provenance, the saved dependency lock, identical device/deterministic flags, and the existing immutable data/feature files. No student-ID plaintext is needed: the public resolved profile is already in each config. `ExperimentRunner(config).run(resume=<trusted last.pt>)` restores optimizer, scheduler, scaler, RNG and loader state; unit tests verify uninterrupted/resumed equality.

AudioCNN uses max100epochs/patience15 and val macro-F1 selection; all fusion/control heads use exactly200epochs/2800updates. Early-stopping was disabled for the compared heads. The AudioCNN is frozen before all head experiments. The source visual CNN is the original Lab2 validation-selected checkpoint, without retraining.

## Unified single-clip inference

`src.data.multimodal_data.clip_features(clip_id)` returns `visual`, `audio`, `mask`, `align` for `model(**sample)`. It rejects cal before opening feature/media files, restores canonical visual slots with explicit validity masks and uses the fixed train statistics. Batched packed `[B,642]` training is unchanged. `build_model(config)` registers video/audio/early/gated via `multimodal_fusion`, and frozen late ensembles via `multimodal_late`; ready-to-load actual late configurations are in `late_model_configs.json`. `late_registry_equivalence.json` verifies bitwise equality to the original functional late predictions for both lambda values and all three modality masks.

Missing-frame canonical windows retain3 detection failures plus69 terminal slots without a decoded frame. No synthetic or repeated image was inserted. Original indices and validity masks prevent those slots entering temporal means. `alignment_metadata_audit.json` records the repair and identical pooled-input SHA256; originals for affected clips remain locally backed up. The single-clip adapter is independently checked against all1320 fixed pooled inputs.

## Chronology, corrections and limits

Exploratory hypotheses were committed before real training. `hypothesis.md` and `E1_validation_selection.json` were committed in `23f41527e22bd35ed26fa3aa728a99b233fa7878` before the main paired runs. Seed1 had already been explored; all main seeds1,2,3 were retrained and are not described as three previously unseen seeds. The original hypothesis is unchanged. All three differences are positive, but the mean is below twice the sample SD: **различие не установлено**.

`execution_failures/` preserves the initial inference-freeze capacity-count failure, the bookkeeping-only correction/reuse scope, original alignment metadata and the descriptive late-ensemble capacity corrections. Corrected counts include both late heads; original values remain in Git. No probabilities, learned weights, selection decisions or paired statistics were modified by these corrections. Lab0–8 scientific outputs are hash-protected.

Teacher actor split remains UNVERIFIED; the user authorized the deterministic actor-disjoint fallback. The conflicting generic `vit_small` profile field is preserved; explicit `cnn_lab2` source precedence was previously authorized. The connected teacher PDF text was read and its Russian prose matched; remote/local raw-byte equality was not claimed after403. Original/A/B pair emotion counts are saved: reused clips and identity/emotion-mixture changes limit causal interpretation. ProtocolB has no Accuracy/F1/ECE/selective-risk target. Noise thresholds are discrete grid comparisons, not assumed monotone cutoffs.

This task produces experiment evidence, figures and Russian counterfactual answers, **not a formal LaTeX/PDF report**.
