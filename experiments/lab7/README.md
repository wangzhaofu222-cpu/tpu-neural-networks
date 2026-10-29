# Lab7 reproducibility and evidence

This task defers all formal Russian/LaTeX/PDF reports and stops beforeLab8. Historical Lab4Q2 remainsBLOCKED,Lab5course split remainsUNVERIFIED,Lab5E2magnitude remainsFAIL,Lab6E1mixing remainsFAIL. The complete unchanged Lab6 is independently verified in `../lab6_verification`:153oldtests,234audit checks,63strictlyloadablecheckpoints and byte-identical CSV/answers/PNG regeneration. No old model is retrained.

## Shared framework and fixed protocol

Use the existing `.conda-env/python.exe` from repository root; no package was added or upgraded. `docs/environment/lab7-requirements.lock.txt` records exact dependencies. All optimization remains inside `src.engine.runner.ExperimentRunner.run`; `src.engine.diffusion.DiffusionState` only supplies the objective, private noise RNG and fixed validation evaluator. Optimizer, scheduler, AMPscaler(disabled),step logging, provenance,DataLoader,checkpoint,globalRNG,resume andpause are shared with previouslabs. Changes in shared modules are scoped to lab7/new registry names; legacy153tests are rerun.

Profile:T1000,beta.0001→.02,M6,rho.1,pi0.02,exact private-ID lab_pick. Defaultwidth256/depth6/time128 retained after actual CPU/GPUbudget tests. Every hidden layer receives a projected transformed sinusoidal time embedding before SiLU; timeMLP usesSiLU too. All comparisons use the same architecture. Main prediction parameterization is epsilon; Lsimple is mean overbatch of theSUM of squared errors across2coordinates,notNLL. DiagnosticsEq7.18 additionally divide by2.

FormaldeviceCUDA(RTX3060Laptop),float32,noAMP,noTF32,2hostthreads; schedules arefloat64. Exact-score sampler usesfloat64 for reference accuracy, from the same underlyingfloat64xT draws;network receives theirfloat32cast. Timing curves are labeled separately bypredictor/dtype. Adam.001,batch512,drop_last,20000updates,constantLR. Fixedbudget finalweights are used; val is monitored without extra endpoint selection. Full endpoints may be partial dataepochs and are inference-only.

EveryRUN also has `schedule.resolved.json`, a numerical companion recording the actual effectivebeta range, terminalalpha_bar, validity, variance convention,tau andDDIMeta where applicable. Original cosine configs retain the same unusedlinearbeta arguments for provenance; they do not define the actualcosinebeta range, whose cap is.999. The companion explicitly distinguishes these quantities without rewriting any originalconfig/checkpointhash. Training/noise-onlyRUNs have no reverse-samplingtau/eta.

Data roles20k/5k/5k seeds701/702/703;cal is empty andnever read. Unequalweights include raremode0. Per-coordinate trainmean/populationstd is fitted only ontrain;val/test/exactscore use the same transform. Exactscore in standardized coordinates has diagonal componentvariancesrho²/scale²; forwardnoise addsidentityvariance. Originalunits are restored beforequalitymetrics. The two rare-boundary datasets are genuinely regenerated/retrained withnewtrainstats atpi0/2andpi0/4.

MMD² is empiricalV-statistic includingdiagonals. h is exactmedian of199990000distinctpairs fromall20000original-unittrainpoints,shared throughoutLab7. Modefrequenciesreference comes from100000targetpoints(seed704),strictdistance<3rho,withoutliers incategoryM. Cov uses referencefrequency/2 perPDF; pi0/2rarecoverage is additionallyreported. Each targethas fivefresh5000true-vstestnoise comparisons(seeds7601..7605),withmeanandsampleSD. E3 teacherdecision uses itsPDFtwo gates(samesign,absmean≥2SD); metricnoise is additional interpretation,not a silentlymodified gate.

## Evidence navigation

- `profile.json`, `teacher_source.json`, `data_protocol.json`: assignedparameters,16pageoriginalPDFhash,rolehashes,trainstats andinverseformula.
- `../../docs/requirements/requirements_lab7.md`: all mandatory clauses,thresholds,optionalitems andfinalstatuses; prompt/PDF interpretations are recorded prospectively.
- `exploratory_hypotheses.md`: E1/E2prospectiverecord inGitbeforetraining.
- `E1.json`: bothfulltrainings,theory,allfixedseedforwardmoments andexplicit invalidterminalT/10exactpredictorcontrol. Rawtrainpointclouds andschedules are in itsdiagnosticRUN.
- `E2.json`: frozenlinear model,all6DDIMK+DDPMfornetworkandexact,onewarmup+5actualtimings each,extra3initialnoisemeasurement,smallestKrule,K1/2samples. PertrajectoryNFEiscounted;all5000samplesaregeneratedinonebatch,soactualcallsperpassmatchNFE.
- `diagnostics.json`:25timesteppercoordinateerrors,irreducible/networkcomponents andfiniteMCcross-term;41²scorefields at4noiselevels withptweighting;full1000DDPMtrajectories perE1schedule.
- `hypothesis.md`, `E3_protocol_lock.json`, `preregistration_evidence.json`: exactE3A/B,uniquechangedfactor,seed1explicitreuse,signedprediction/falsifier,realGit chronology. NeverrewriteafterE3.
- `E3.json`: threepairedseeds,allmodefrequencies,p0,Cov,Dmode,MMD² andindependentpairedstatistics.
- `supplements.json`: twofullrare-targettrainings,onefull10T20000updateQ1training,zeroepsilonvariancecontrol. Q1actuallymeasures5000samplesDDPMtime,not just aformula.
- `counterfactual_answers.json/md`: fivequestions withrealnumbers/calculations/RUN_IDs;Q4additionallyevaluatescosineDDIM5;Q5usesactualCov1runorhonestlimitation.
- `figures/`, `figure_provenance.json`, `figure_QA.json`: SVG/PNGchartsderivedonlyfromsavedrecords,andvisualreview.
- `final_audit.json`, `checkpoint_inventory.json`, `raw_artifact_inventory.json`, `execution_statistics.json`: independentintegritychecks,teacherthresholdoutcomes,alllocalartifacthashes/counts/timing.

## Commands and preservation

Fromrepo root:

```powershell
.\.conda-env\python.exe -X utf8 -m reports.tools.lab7_execute e1
.\.conda-env\python.exe -X utf8 -m reports.tools.lab7_execute e2
.\.conda-env\python.exe -X utf8 -m reports.tools.lab7_execute diagnostics
```

Thecontrollerpersistentindexreusescompletedformaltrainings. Formalrunningrequirescleancommittedsource/config. Do not deletecompletedindices orchange20kbudget. E3preregistrationmustbecommittedbefore`reports.tools.lab7_e3`;newreproductionsneedafreshoutputnamespaceandtheirownprospectivechronology. Historiccheckpointsmustusetheirrecordedsource/config/data;donotbypasssourcehashguards.

`experiments/lab7/PAUSE` requestsstopatacompleteepochboundary. Resume via the existing runner/`src.train --resume <checkpoint>`withtheexactsameconfig/source;fullmodel/optimizer/scheduler/scaler/globalRNG/loaderRNG/private training-noiseRNG arerestored. CPUandGPUtestscompareuninterruptedversusresumedstatesbitwise. Finalpartialepochcheckpointdoesnotclaimmid-epochresume.

After allrawresults exist,analysis canberecomputedwithoutoptimization:

```powershell
.\.conda-env\python.exe -X utf8 -m reports.tools.lab7_artifacts
.\.conda-env\python.exe -X utf8 -m pytest -q --junitxml=docs/environment/tests-lab7-final.xml
.\.conda-env\python.exe -X utf8 -m reports.tools.lab7_audit
.\.conda-env\python.exe -X utf8 -m reports.tools.lab7_finalize
```

Theauditdoesnottrain;itrecomputesmetricsandactuallyreplayssamplersfromsavedweights/initialnoise/RNGseeds. Largedata,weights,rawsamples,scorefieldsandtrajectoriesstaylocallywithSHA256inventories;smallpassports/configs/logs/tables/plots/tests/auditsaretrackedinGit. Preserve localartifacts. ScientificFAILisnothiddenbyintegrityauditPASS.
