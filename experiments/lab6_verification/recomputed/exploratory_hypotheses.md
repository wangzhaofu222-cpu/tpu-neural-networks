# Lab6 prospective E1/E2 exploratory protocol

Both experiments use seed1 and the exact dataset roles20k/5k/5k generated with601/602/603, unstandardized coordinates. No cal role. Every model executes5000Adam updates,lr.001,batch256,constant schedule,fullprecision. Fixed-budget final weights are evaluated; no hidden selection among training budgets. ExperimentRunner from Lab0 remains the sole training loop and stores full model/optimizer/scheduler/scaler/globalRNG/loaderRNG/replay state.

## E1

Train exactly one EBM on two_blobs, hidden128×3, ReLU,tanh residual, external quadratic s3,eta=.02,K50,T1,replay10000,restart.05. All four evaluation eta [.005,.02,.08,.5] use the same final weights, fixed standardNormal5000initialpoints and common random stream seed7001; reflecting domain[-15,15]². Center mixing is measured separately using5000chains initialized exactly at center0,seed7101,not confused with global initial noise. Record all51trajectory states,DK,energycurves,valid mode changes,outlier-label changes,finaldomainfraction and pre-reflection outside proposal fraction.

Prediction: eta→0 makes displacement vanish at fixedK (6.6). Too-small steps underexplore, while larger steps travel farther but introduce discretization bias. The ideal quadratic chain variance is1/(1-eta/2), not1 (6.7): [1.002506,1.010101,1.041667,1.333333]. Its maximum is≥1.2. Eight independent chains each execute10000burn-in+200000retainedsteps. Seeds8101..8104 in eta order; wrong-noise control seed8105 keeps sigma=sqrt(2*.02) while eta=.5,Teff=.04,variance=.053333. All trajectories/innovations are saved. A compiled scipy lfilter evaluates the exact finite AR1 recurrence and is crosschecked against explicit autograd-gradient updates, not replaced by sampling an analytic stationary Gaussian. Error>5% is preserved, not repaired by seed shopping.

Predict learned energies at both centers lower than midpoint, but the bounded residual can only change relative energy by at most2. With quadratic scale3 it may not concentrate sufficiently on width.3blobs, so poorCov/MMD² and insufficient mixing are plausible model limitations. No extra regularization (coefficient0), no altered residual bound or scale will be introduced to improve outcomes. The tail in(6.3) is an unnormalized integral bound; do not call it an exact tail probability.

Workingeta: smallest stable value with noNaN,finaloutside≤.05,center→othermode endpointfraction≥.2. Otherwise choose stable minimumMMD², ties byeta order, and explicitly state K50 did not meet the mixing criterion. PDF defines MMD² using fixedtest, so this exploratory fallback uses that two_blobs test-defined statistic openly; it does not select E3 on eight_gaussians outcomes. 'Too large' refers to the strongest theoretical discretization bias, not an invented claim that it diverges. Main empirical conclusions may contradict these predictions.

## E2

Train four independent flows on two_blobs, Kf2/4/8/16,eachs/t twohidden64ReLU,cap4,small nonzero random output weights at initialization. ExactlyKf−1 coordinate swaps; output is data→noise with signed logdet. All5000updates use the samebatchprotocol/seed1. ValNLL is monitored during training; the endpoint weights are used consistently. Record per-pointvalNLL,itsmean/sampleSEM,5000sample MMD²/Cov,samplingtime,parametercount. Inversion of converted learned weights is measured infloat64 on all5000testpoints and100fixednorm20points,seed6201;explicit autograd Jacobians use first100testpoints.

Prediction: extra coupling layers initially reducevalNLL, then improvements can saturate or become optimization-limited. Betterlikelihood need not improve sampleMMD² (6.9,6.13). No assumption is made that inverse error necessarily grows on norm20; report actual values and any thresholdFAIL.

PDF does not define which SEM to use for an adjacent-depth difference on sharedvalpoints. Prospectively use sampleSD of paired per-point NLL differences /sqrt5000; improvement is significant iff mean decrease>2SEM. Also retain each depth's ownSEM. Choose the smallestdepth whose next improvement is not significant; if all three gains are significant choose8. Never use test to selectdepth. An improving16layer test score cannot override this rule.

## Shared metric estimators

Grid120×120cellcenters,spacing.25,area.0625. MMD² is empiricalV-statistic includingdiagonals, with allpairs kernel means, same everywhere. Bandwidth median of all1999000distinctpairs of2000trainpoints sampledwithoutreplacement,seed604,oncepergoal. Covthreshold uses exact nonoverlapping3sigma-disk probabilities under the knownmixture. MainDmode is the plug-in two-sample distance to fixed5000testmodefrequencies including−1; distance to exacttargetfrequencies is separatelyreported. Fivefresh5000target batches seeds6101..6105 vsfixedtest define each metric's noise baseline. Allchoices are fixed before anyformal E1/E2 evaluation.

E3 target is assigned eight_gaussians. E3 primary metric/order/prediction and allowed EBM width matching will be committed separately after E1/E2 selections, before any E3training. Noformalreport orLab7 is in scope.
