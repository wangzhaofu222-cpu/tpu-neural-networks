# Lab3 experiment evidence (no formal report)

Lab2 was audited independently before Lab3. See ../lab2_verification/audit.json; 70oldtests and38checkpoints passed; original artifacts unchanged.

Profile: Copy; L0=128;2PreNormblocks;d128,h4,FFN256;B32;AdamWwd.01;fixedlr.003;2000steps. LR chosen once by sinusoidal final valL0 only. Canonical explicitfloat32 training has no mask,padding,dropout.

## E1

| PE | M128 | M256 | M512 | M1024 | ER256 | ER512 | ER1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
|none|0.058593750|0.041473389|0.030776978|0.024765015|notplotted|notplotted|notplotted|
|absolute|0.125671387|0.103271484|0.078186035|0.055778503|0.821758|0.622147|0.443844|
|sinusoidal|0.469177246|0.024963379|0.021469116|0.016548157|0.053207|0.045759|0.035271|
|rope|0.110717773|0.015014648|0.015151978|0.015937805|0.135612|0.136852|0.143950|

Both validation thresholds PASS: noPE=.057861328125 <=.065625; bestPE−noPE=.40338134765625 >=.20. P1at256=.027008056640625 versusP2=.103271484375. First longer failure: absolute1024, sinusoidal256, RoPE256; noPE already below effective threshold atL0. Absolute longest measured effectiveL512, sinusoidal/RoPE128. NoOOM. ER for noPE retained raw but suppressed in figure to avoid overinterpreting its frequency-based near-baseline performance.

## E2

| Backend | L | Forward ms | Trainingstep ms | Actual peak MiB |
|---|---:|---:|---:|---:|
|explicit|64|2.252347|8.738990|68.548828|
|explicit|128|2.461313|9.263983|71.023438|
|explicit|256|2.283307|9.669897|80.344727|
|explicit|512|3.198130|9.318210|114.883789|
|explicit|1024|2.975173|9.175557|243.961914|
|efficient|64|1.840380|8.365243|68.306641|
|efficient|128|1.757223|7.599493|69.984375|
|efficient|256|1.798300|7.928713|73.851562|
|efficient|512|1.685950|7.405270|81.929688|
|efficient|1024|2.834573|9.242893|98.085938|

Raw memory interval checks FAIL for both backends. The PDF§9.2 alternative decomposition is completed separately; rawFAIL values are not rewritten. At longestL explicit memory fit is66.7%quadratic,5.9%linear,27.4%constant; efficient1.4%quadratic,31.0%linear,67.5%constant. Forward latencies are not monotone and cannot establish a pure quadratic timing regime at this fixed small width. Five same-processpilot profiles remain excluded from primary fitting. ForcedFlash is not compiled on this Windowsbuild; forcedEFFICIENT_ATTENTION is supported and measured independently. Both primary curves arefloat16, with10warmups and30rawrepeats, freshprocess percondition. Actualallocatorpeak is distinct from theoreticalmatrixbytes/largesttensor shape audit.

## E3

Preregistered commit: 4d6e559acec22b6af2d78e8ef96ff452a77b1d4c. A=RoPE,B=absoluteP2,L*=256. Seeds1,2,3; seed1 reused. Deltas=[-0.0882568359375, -0.07147216796875, -0.056610107421875]; mean=-0.072113037109375; sampleSD=0.01583309481361658. Allnegative andabsmean>=2SD: difference established, preregisterednegativehypothesis supported. Selection on exploratoryE1/seed1 is disclosed; newseeds2/3 were unseen. See hypothesis.md, selection.json, preregistration_evidence.json andE3.json.

## Diagnostics, tests and storage

All256examples atL128 and256 for the bestval sinusoidal model; raw per-example/head/query statistics are attention_raw.pt inside the two diagnosticRUNs, summaries in attention_analysis.json andrawattention.json. Secondblock mean targetmass falls .222375 -> .008194. These are routing diagnostics, not causal explanations. All five counterfactual questions, including newwidth256timing, are in counterfactuals.json withRUN_IDs.

Finaltests: 88passed,0failed/errors/skips. Sixscientificfigures inspected visually (figure_qa.json). Formaltraining:10executions (3LR+3newE1+4newE3),20000updates,293.600067000s. Including completed smoke/budget:12executions,295.345164300s; onefailedinitialsmoke retained separately. Profilingtraining-step repeats are not counted as scientifictraining. Savedcheckpoints24(formal20), SHA256index incheckpoint_index.json. Weights remainlocal inruns/lab3/*/checkpoints and are ignored byGit.

Primaryevidence: results.json, audit.json, ledger.json, E1.csv, E2.csv, E3.csv, E2_analysis.json, boundaries.json, source_compatibility.json. TotalindexedRUN_IDs=56; aliases are not extra training runs. Cal was generated but neveropened; it remainsreserved untilLab8. NoLaTeX/PDFreport created ormodified; Lab4notstarted.

## Regenerate derived analysis

Run from projectroot with existing localrecords/checkpoints:

```powershell
.conda-env/python.exe -X utf8 -m reports.tools.lab3_results
.conda-env/python.exe -X utf8 -m reports.tools.lab3_audit
```

These commands readrawresults and updateonlyLab3derivedtables/figures/audit. Do notrun the Lab2generator against originalLab2records. Historical model recovery uses eachmanifest sourcecommit; E1-to-E3 activeexplicittrainingmath is unchanged, fullsource-diff interpretation saved.
