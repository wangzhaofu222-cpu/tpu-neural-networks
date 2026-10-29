# Lab6 E3 prospective primary comparison

This immutable hypothesis and its resolved configuration lock must be committed to actual Git BEFORE any E3 model training. It is created after E1 and E2 completed, not reconstructed from E3 outcomes. Future interpretation belongs in a separate analysis file.

Q = MMD² (saved field MMD2), empirical RBF V-statistic including diagonals.
A = EBM; B = Flow.
Delta_s = Q(EBM,s) - Q(Flow,s).
Expected sign = positive (Flow has lower sample-distribution discrepancy).

## Fixed protocol

Target = eight_gaussians, original unstandardized coordinates, Omega=[-15,15]². Train20000seed601,val5000seed602,test5000seed603; no cal. All six models execute5000Adam updates,lr.001,batch256,constant scheduler,float32,CPU2threads,seeds1/2/3. All paired arms share dataset,mini-batch order by a private loader generator,optimizer-step budget,evaluation code and5000generation samples. Each pair's generation uses seed7000+seed for standardNormal initial noise; EBM continues with its Langevin noise, Flow applies its inverse. Fixed-budget endpoint weights are evaluated. No checkpoint/test-driven tuning.

EBM keeps three hidden ReLU layers, external s3 quadratic, bounded tanh residual, replay10000,restart.05,K50,T1,eta=0.02 selected by E1. Energy regularization coefficient0. E1 mixing criterion satisfied=False; selection rule: fallback stable minimumMMD2; K50 mixing criterion not met. The inherited step is not assumed to mix all eight modes.

Flow keeps E2-selected Kf=8,Kf−1swaps,c=4.0,two separate s/t networks perblock,each2hidden64ReLU. Its exact testNLL remains distinct from EBM grid estimates.

The permitted capacity adjustment changes only EBM hidden width from128 to185. EBM parameters=69561;Flow parameters=69648;gap over the smaller count=0.00125070,≤20%. No other depth,Kf,eta,K,temperature,target or optimizer-budget change is allowed. Exact six configurations and their hashes are in E3_protocol_lock.json.

## Prediction and falsification

Expect positive meanDelta: a bounded residual can change energy by at most2 while the s3 Gaussian prior is broad relative to sigma.2 target modes. Short-run K50 Langevin may retain initial/noise/replay-related biases. A trained flow fits exact density by(6.9) and samples by explicit inversion, so we expect smaller MMD². This is a model/protocol prediction, not proof that likelihood universally implies sample quality or that an observed gap has only one cause.

The direction is contradicted by nonpositive meanDelta. Inconsistent signs or failure of the teacher statistical/noise gates leave the difference unestablished even if the mean is positive. Declare 'различие установлено' only if all three nonzero differences have one sign,abs(mean)>=2*sampleSD(ddof1),AND abs(mean)>=MMD² noise mean from five fresh5000target-vs-fixedtest comparisons. Otherwise declare 'различие не установлено'. Show everyDelta,mean,SD andnoise. Do not replaceQ,reverseA/B,selectseeds,changebandwidth or tune width after observing outcomes.

The bandwidth is determined once from fixed2000trainpoints,seed604. The noise baseline follows seeds6101..6105, with all raw samples saved. Dmode andCov are secondary descriptors, not substitutes for the preregistered primary test.

## Midpoint diagnostic and scope

Nearest-center pair is chosen solely by geometricdistance,rounded tie-breaking by original center indices. Record true logdensity,Flow exact logdensity andEBM Omega-grid estimated logdensity at its midpoint,plus the full201pointline and densitymaps,for allthree seeds. Report excess logdensity continuously; do not presume a visible bridge or attribute every gap solely to invertibility. Positive density everywhere is not itself evidence of a large visible bridge. No further model training/tuning follows E3 comparison. Noformalreport orLab7.
