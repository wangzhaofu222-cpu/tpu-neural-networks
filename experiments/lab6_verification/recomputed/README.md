# Lab6 reproducibility and evidence

No formal report is generated in this task. No Lab7 work is authorized. Historical Lab4Q2 remainsBLOCKED;Lab5course split remainsUNVERIFIED;Lab5E2 magnitude prediction remainsFAIL. Lab5 was regenerated in `experiments/lab5_verification`,130tests and625checks passed,274formal/286allcheckpoints retained;5450files are protected by SHA256.

## Environment and shared harness

Use the existing `.conda-env/python.exe` from repository root. No dependencies were upgraded/added. Exact environment lock is `docs/environment/lab6-requirements.lock.txt`. CPU2threads,float32 is the formal protocol; flow inverse/Jacobian diagnostics convert learned weights tofloat64. All optimization uses the original `src.engine.runner.ExperimentRunner.run` with scoped generative adapters in `src/engine/generative.py`. It is not a second trainer. The shared optimizer,step counter,DataLoader,logs,provenance,RNG,checkpoint/resume andpause mechanisms are reused. The EBM adapter additionally stores the complete replay buffer and private sampling RNG.

All models execute5000Adam updates,lr.001,batch256,constant schedule. Train uses shuffled fixed20000points with drop_last=True to keep every update256;examples cycle across dataepochs. Final5000step weights are evaluated consistently. ValNLL is monitored eachdataepoch for Flow, but no unrequested endpoint/budget search is performed. E2 depth is selected by val only. EBM valmeanenergy is explicitly not NLL and does not select weights. EBM loss is positive meanenergy minus negative meanenergy with only the negative input detached. Regularization coefficient0.

## Evidence navigation

- `profile.json`,`teacher_source.json`,`data_protocol.json`: private-input-derived public profile, original15pagePDFhash, allfourmixture definitions androle tensor hashes. RawID never belongs in publicconfig.
- `../../docs/requirements/requirements_lab6.md`: mandatory/optional requirements, prospective statistical clarifications and final statuses.
- `exploratory_hypotheses.md`: E1/E2prospective protocol, committed before formaltraining.
- `hypothesis.md`,`E3_protocol_lock.json`,`preregistration_evidence.json`: E3realGitpreregistration,immutable numerical choices,exactresolvedsixconfigs andhashes. Do not overwrite them.
- `run_index.json`: source training/evaluation/quadraticRUN_IDs.
- `E1.json`,`E1_eta.csv`,`E1_quadratic.csv`: onefrozenenergy,allfoursteps,centerprobe,mixingfallback,fullquadraticvariance checks andwrongnoisecontrol.
- `E2.json`,`E2_depths.csv`: allfourdepths,pairedvalSEMdecision,exactNLL,samples,invertibility andfarfield diagnostics.
- `E3_capacity_match.json`,`E3.json`,`E3_models.csv`,`E3_pairs.csv`: allowedwidthadjustment andallthreepairedseedresults. EBMgridestimate is not FlowexactNLL.
- `metrics_protocol/*.json`,`noise/*.json`: one train-only bandwidth pertarget andallfiveactualnoise comparisons per experimental target.
- `counterfactual_answers.json/md`: allfive questions,measurements,Chinese explanations andRUN_IDs.
- `figures/*.svg/png`,`figure_provenance.json`,`figure_QA.json`: raw-record-derived scientificplots,vector exports andvisualQA.
- `final_audit.json`,`checkpoint_inventory.json`,`execution_statistics.json`: independent numerical/provenance verification,localcheckpoint hashes andactualruntime/counts.

## Data and metric semantics

Four equal Gaussian mixtures use exact teacher centers/sigmas. Roles are20000/5000/5000 samples,seeds601/602/603;cal is not created;coordinates are not standardized. The samplemode is nearestcenter only within3sigma;otherwise−1. Fixed2000trainpoints(seed604) set the median pair-distance RBF bandwidth once pergoal. MMD² uses the empirical V-statistic includingdiagonals;allpair terms are computed blockwise. Dmode uses plug-in fixedtestmodefractions including−1,consistently for models andnoise baselines;distance to exacttargetfractions is separatelyreported. Cov uses exacttargetdisk probabilities. The fivefresh5000targetvsfixedtest comparisons use seeds6101..6105.

EBM grid uses120²cellcenters in[-15,15]²,spacing.25,area.0625,logsumexp normalization. Its normalized_on_omega flag is true;logdensity is an estimate. Eq6.3 tail≈.002869 is an unnormalized integral upperbound. Division by estimatedZ is not represented as an exact probability bound. Flow log_prob is exact and its plotted grid is not renormalized toOmega.

Normal generation starts at standardNormal noise. E1eta conditions share a noise stream and a frozen checkpoint. A separate5000chain centerstart diagnostic measures endpointmode mixing;outlier label transitions are not falsely counted as validmode switches. Boundaryreflection handles repeated crossings;rawoutsideproposal rate is separate from reflected finaloutside rate.

Quadratic validation saves8×210001states and8×210000independentGaussianinnovations percondition. SciPy lfilter executes the finite AR1 update exactly in compiled arithmetic;it does not draw from a theoreticalstationaryGaussian. The implementation is crosschecked against explicitautograd-gradient updates. Every chain discards10000updates and retains200000. The wrong-noisecontrol deliberately keeps sigma fromsecondeta while changingdrifteta to largest. This controlled mismatch does not add a freesigma argument to the main sampler.

## Commands and recovery

```powershell
.\.conda-env\python.exe -X utf8 -m src.train --config configs/lab6.yaml --set experiment.purpose=formal
```

Every call creates a new RUN_ID. Use `--resume <RUN>/checkpoints/last.pt` only for a complete dataepoch checkpoint with exactsame source/config/data/split;model/optimizer/scheduler/scaler/globalRNG/loaderRNG/replayRNG are restored. The finalpartialdataepoch checkpoint at5000updates is for inference,notan unsupportedbudgetextension. `experiments/lab6/PAUSE` triggers a recoverable stop at a completeepoch boundary. Never overwrite a RUN or skip the sourcehash guard. Unit tests compare uninterrupted andresumed model/optimizer/RNG/replay states bitwise.

Serial stages: `-m reports.tools.lab6_execute e1`,then`e2`;create prospective E3files with`-m reports.tools.lab6_preregister`,commit actualGit,andonlythenrun`e3`. The controller reuses its persistent completedindex and never repeats a validformaltraining run unnecessarily. Do not deleteindices to fake a fresh experiment. New full reproductions should use an isolated clean checkout/outputnamespace and preserve prospective chronology.

Read-only model verification (nooptimization):

```powershell
.\.conda-env\python.exe -X utf8 -m pytest -q --junitxml=docs/environment/tests-lab6-final.xml
$env:MPLCONFIGDIR='D:\神经网络\tmp\mplconfig'
.\.conda-env\python.exe -X utf8 -m reports.tools.lab6_artifacts
.\.conda-env\python.exe -X utf8 -m reports.tools.lab6_audit
.\.conda-env\python.exe -X utf8 -m reports.tools.lab6_finalize
```

The audit replays finite samplers and recalculates metrics from trusted checkpoints/rawrecords without optimizer updates. Regeneratedfigures can differ only in SVGcreationmetadata;scientifictables must reproduce. Large localweights/samples/grids/trajectories/innovations are ignored byGit but their hashes are inventoried. Configs,metrics,plots,logs,RUN_IDs,scientificfailures andaudits are tracked. Preserve local artifacts;do not replace missingweights with fabricated results.


## Final independent regeneration

`regeneration_audit.json` confirms byte-identical regeneration of all five CSVs, both counterfactual answer files and eight PNG previews into `verification/regenerated/`, with no training. SVG creation timestamps are not compared. `reports.tools.lab6_regeneration_check` intentionally refuses to overwrite an existing regeneration directory; preserve the verification evidence and use a fresh namespace for another reproduction. `../lab5_verification/all_checkpoints_loadability.json` independently verifies all286 old checkpoints, including budget runs. Durable console logs are in `logs/`.

`requirement_status.json` separates scientific observations from procedural checks: OBS01 is FAIL because mixing never reaches20%; E108 is PASS because the specified fallback was applied faithfully. Alleta Cov=0 is disclosed rather than reclassified as good sampling. The independent audit PASS means measurements/provenance were verified, not that every scientific threshold was reached. Final audit and regeneration do not execute optimizer updates.
