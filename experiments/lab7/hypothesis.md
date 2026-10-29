# Lab7 E3 prospective hypothesis

E1/E2 and diagnostics are complete. This file and E3_protocol_lock.json must be committed in real Git before new E3seed2/3 training/evaluation. Seed1 is explicitly exploratory reuse,not retrospectively labeled unseen.

A={"schedule": "linear", "sampler": "ddpm", "K": 1000, "eta": null}; B={"schedule": "linear", "sampler": "ddim", "K": 50, "eta": 0.0}. Sole factor: reverse sampler configuration. All training conditions and the actual frozen linear model are identical within eachseed. DDPM uses posteriorvariance and allTsteps; DDIMeta0 uses the lockedK. Changing the sampling algorithm includes its step-grid by definition (the example explicitly permitted in PDF§9.3),not a second training factor. TargetM6,rho.1,pi0.02; fixedroles701/702/703; train-onlystats; architecturewidth256/depth6/time128/SiLU; Adam.001,batch512,20000updates; no clipping; samebandwidth/reference;5000samples; commonxTseed7400+s; DDPMzseed7500+s. Seeds1/2/3 control initialization,batches,t,epsilon,sampling. No seedselection.

Q=Dmode,Delta_s=Q(A,s)−Q(B,s). Predicted sign=-1, based on exploratoryDmodeA=0.05625,B=0.09245. Eq7.16's stochastic posteriorvariance and Eq7.17's deterministic coarse reverse integration can distribute mode masses differently; Eq7.19 networkerror contributes to both. This is a conditional prediction on this assigned target,not a general guarantee that either sampler is superior.

Prediction is supported only if allthree differences have the expected nonzero sign andabsmean≥2sampleSD, following PDF(7.21). Opposite sign or a teacher decision 'различие не установлено' does not support the main prediction. Metricnoise is independently reported anddifferencesbelownoise are not interpreted as robust performance differences,without replacing the PDFtwo-condition decision.

Seed1 reused exactly: A_RUN_ID=deed2ad27ea6dc9b82a2-9bde9515fc6f4be4b836698669d64ef6; B_RUN_ID=a1bee33a693a2f0588bb-59d9b1c17aab43d9b3a85c3b6e9aa76d; source=runs\lab7\14955e25e337bf6ddba3-82af941555d947d2bf1c02947758acaf. Seeds2/3 use two newlytrained linear models,each shared by A/B. E3 rarityboundaries train two NEW A-targetmodels atpi0/2andpi0/4,seed1,sameAtraining/sampler; do not merely retest oldweights.

Immutable source selections: E1_SHA256=2c1521bcfaadf3c3cc2019aa07d221ec4ec6b03510eeb7ea43979092b3688511; E2_SHA256=ec13acf8e5dda5e91e8364a2f4ffa5b26d20bda43aeea619eefc25a3a19abc58. Do not revise this hypothesis after E3 results; add separate analysis instead.
