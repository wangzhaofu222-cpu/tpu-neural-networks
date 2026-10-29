# Lab6 mandatory requirement matrix

Original source: lab6_ebm_normalizing_flows_1010.pdf, revision4,15pages, freshly read all pages; formulas visually checked. Status vocabulary: PASS/FAIL/BLOCKED/NOT_APPLICABLE/IN_PROGRESS. No formal report; stop before Lab7.

| ID | PDF original requirement | Implementation | Verification/evidence | Status |
|---|---|---|---|---|
| A01 | User prerequisite: independently verify Lab5 without retraining or rewriting | experiments/lab5_verification;130tests;625checks;274formal/286all checkpoints | copied regeneration and protected5450historical hashes | PASS |
| A02 | Preserve historical E2 magnitude FAIL, Lab5 course split UNVERIFIED, Lab4Q2 BLOCKED | original records remain immutable | final protected-file audit | PASS |
| P01 | §8 lab_pick exact original private SID; target,eta,c | existing src.profile_gen; configs/lab6.yaml | independent SHA256 slices | PASS |
| D01 | §7,(6.16): four equal isotropic Gaussian mixtures, exact centers/sigmas, no standardization | distributions.py | midpoint−21.65±.01,center modes,sampling tests | PASS |
| D02 | §7,§9.1:20k/5k/5k seeds601/602/603; no cal; domain[-15,15]² | deterministic dataset adapter | fingerprints,role isolation,regeneration | PASS |
| D03 | §9.1:sample/log_prob/mode_id/domain API; distance>3sigma returns−1 | distribution registry | center/outlier tests | PASS |
| M01 | §6.4,(6.12):120² cell centers,spacing.25,area.0625; Zhat,phat,Dgrid | density.py | grid mass,logsumexp,direct oracle | PASS |
| M02 | §6.4:normalized_on_omega flag; EBM grid estimates distinct from exact FlowNLL | typed evaluator results | no fake energy.log_prob; unit tests | PASS |
| M03 | (6.13):5000generated vs5000test;Cov,Dmode including−1,MMD² RBF | density_metrics.py | explicit estimators,counts/kernel tests | PASS |
| M04 | §6.4:median pair distances fixed2000train;one bandwidth pertarget,selection seed saved | bandwidth protocol | exact pdist oracle and cache fingerprint | PASS |
| M05 | §9.1:five independent fresh-target vsfixedtest noise comparisons per experimental target | baseline RUNs | allfive raw samples/metrics and mean | PASS |
| B01 | (6.2),§9.2:external quadratic s3 +3hidden128ReLU bounded tanh residual | ebm.py | scalar shape,lower bound tolerance1e−5 | PASS |
| B02 | (6.6):autograd.grad input;noise²=2etaT;no free sigma argument;T1 | samplers.py | parameter/gradient immutability and sigma tests | PASS |
| B03 | §6.2,§9.2:reflection not clamp;unbounded variance mode;full x0..xK | sampler | multiple-boundary reflection and trajectory tests | PASS |
| B04 | (6.8):10k random-replacement replay;restart.05 standardNormal;fullbuffer and RNG checkpoint | shared runner replay hook | bitwise state and continuation equality | PASS |
| B05 | (6.4):detach negative input only;two-phase parameter gradient | shared runner objective | analytical gradient error≤1e−5;wrong detach fails | PASS |
| B06 | §9.2:Adam.001,batch256,5000steps,K50,T1,secondeta inE1 | ExperimentRunner.run | actual5000steps;fulloptimizer/RNG checkpoints | PASS |
| B07 | §9.2:positive/negative energy,loss,gradnorm,DK,trajectoryenergy,mode switches | steps.jsonl+sampler diagnostics | actual raw records | PASS |
| F01 | (6.10–11),§9.3:independent s/t each2hidden64ReLU; c*tanh(raw/c) | flows.py | shared forward/inverse weights and scale bounds | PASS |
| F02 | §9.3:Kf layers,Kf−1 swaps,zero swaplogdet | Flow composition | counts/identity/permutation tests | PASS |
| F03 | (6.9):data→noise,exactlog_prob,Normal base,reverse sampling | Flow | signed determinant,probability and batch-independence tests | PASS |
| F04 | §9.3:Adam.001,batch256,5000steps;valNLLmonitor | shared runner objective/evaluator | actualsteps/metrics/checkpoints | PASS |
| H01 | §10:E1/E2 exploratory hypotheses seed1 before experiments | exploratory_hypotheses.md | real Git ancestry/timestamps | PASS |
| E101 | §10.1:one two_blobs EBM frozen for ALLeta | E1 source checkpoint | identical checkpoint SHA for allsamplerconditions | PASS |
| E102 | §10.1:energy/normalizeddensity/true maps,samples;two center/mid/testenergies | evaluator/plots | raw grid and energy measurements/RUN_ID | PASS |
| E103 | (6.3):R14 unnormalized tail integral bound≤.01,about.0029 | tail_bound | analytical double oracle | PASS |
| E104 | §10.1:both trained-center energies below midpoint | acceptance | preserve real FAIL if violated | PASS |
| E105 | §10.1:allfoureta,T1,K50,MMD²,Cov,DK,outside/stability,trajectories | evaluation RUNs |5000samples,full51states;pre-reflection proposals separately | PASS |
| E106 | §10.1,(6.7):each eta8chains,burn10k,200kdraws;variance relativeerror≤5% | quadratic finite-chain evaluations | allinnovations/states;crosscheck genericautograd kernel | PASS |
| E107 | §10.1:fixed sigma fromsecondeta onmaxeta,Teff andactualtheory;≤5%error | wrong-noise control RUN | full8×210k recurrence,notcontinuousT variance | PASS |
| E108 | §10.1:mineta stable,outside≤5%,≥20%center→othercenter;else stable minMMD² and mixing failure stated | frozen selection rule | validation from center-start chains,not sampleinitialization | PASS |
| E109 | §10.1:too small/working/too large;maxeta theoryvariance≥1.2 | E1 interpretation | actual values and registeredeta grid | PASS |
| E201 | §10.2:Flow depths2/4/8/16 two_blobs same5000steps,batch,lr,seed,c | E2 runs | completefourrows/configs | PASS |
| E202 | §10.2:valNLL±SEM,MMD²,Cov,params,sampling5000time,testepsilon | evaluation | saved pointwisevalNLL,raw samples,timings | PASS |
| E203 | §10.2,§12:float64 test medianepsilon≤1e−5;autograd logdeterror≤1e−5 | invertibility.py | alltest roundtrips;explicit2Dautograd Jacobians | PASS |
| E204 | §10.2:100points norm20epsilon;disclose no growth if absent | far-field diagnostic | fixedseed100circlepoints | PASS |
| E205 | §10.2:first depth whose next gain≤2SEM;allgains clear→8 | preregistered val-only selector | paired per-point val-difference SEM;both means±SEM also saved | PASS |
| E301 | §10.3:profiletarget;keepeta,K,Kf,c,steps,seeds;onlyclass/allowedwidth differ | E3locked configs | capacitymatching metadata | PASS |
| E302 | §10.3:parametergap≤20%,adjustonlywidth,notdepth/Kf | paramcounter | requireabsdiff/min(counts)≤.2 (also maxdenominator passes) | PASS |
| E303 | §10.3:chooseQ=MMD² orDmode,A/B,expected sign,falsifier;actual Git BEFORE E3 | hypothesis.md | immutable normalized Gitblob/ancestor/starttimes | PASS |
| E304 | §10.3:sixformal models,seeds1,2,3,eachfull5000steps | ExperimentRunner | pairedmatrix,completecheckpoints | PASS |
| E305 | §10.3:params/train&sampletime/Cov/Dmode/MMD²;Flow exacttestNLL;EBM Dgrid/Zhat/tail/flag | common evaluator | recompute from models/rawsamples | PASS |
| E306 | §10.3:nearest-centers midpoint,true/Flowexact/EBMgridlogdensity;segmentmaps/bridge | midpoint diagnostic | recordmethod,coordinates,allthree seeds | PASS |
| E307 | (6.15),§10.3:threeDeltas,mean,sampleSD,same sign,absmean≥2SD AND ≥metricnoise | pairedstatistics | independent arithmetic;preserve nonestablishment | PASS |
| C01 | §11.1:two minimumeta DK andfixedK limit | counterfactuals | real numbers+RUN_ID+scope | PASS |
| C02 | §11.2:maxeta quadraticvariance vs learnedstability/Cov/MMD² | counterfactuals | real data,not theoretic speculation | PASS |
| C03 | §11.3:wrong-noiseeta/sigma/Teff/measured&theoryvariance;continuous vsdiscrete | counterfactuals | actualcontrolRUN | PASS |
| C04 | §11.4:twoFlowdepth NLL/MMD²,likelihood vssamples | counterfactuals | actualE2numbers | PASS |
| C05 | §11.5:three midpointlogdensities;lowtrainingenergy not enough | counterfactuals | actualmaps/values/RUNs | PASS |
| T01 | §12.1:energyshape/externalquadratic/lowerbound≤1e−5 | test_lab6_generative.py | actualpytest | PASS |
| T02 | §12.2:noise relation/no freesigma/full eta.05 chainvariance≤5% | tests | full8chains10k+200k,not smoke | PASS |
| T03 | §12.3:samplerweightsunchanged/parametergradients/detachnegativeinputonly | tests | maxerror≤1e−5 and deliberatelywrongimplementationrejected | PASS |
| T04 | §12.4:float64norm≤5roundtripmedian≤1e−5;constant−1 andrandomnonzeros autogradlogdet≤1e−5 | tests | signedJacobian construction with autograd.grad | PASS |
| T05 | §12.5:identity≤1e−6;selfinverseswap;single−1layerdensity;batchindependence | tests | actualpytest | PASS |
| T06 | §12.6:normalized_on_omega,no fakeexactlogprob,bitwisereplaycheckpoint | tests | actualpytest | PASS |
| T07 | User:deterministicdatasets/roles/RUNunique/optimizerRNGresume/metrics/capacity/statistics/Gitprereg/regression | tests | fullold130 andnewtests | PASS |
| DEL | §14:allRUNs/rawdata/configs/weights/RNG/logs/trajectories/tables/figs/tests/finalaudit/Git | experiments/lab6,runs/lab6 | independentregeneration/protectedhistory | PASS |
| RPT | User explicitly defers formal Russian/LaTeX/PDF report andLab7 | scope | no report generated,no Lab7 execution | NOT_APPLICABLE |
| X01 | §15.1 optionalchainlengthsweep | not required | not executed | NOT_APPLICABLE |
| X02 | §15.2 optionalOODlikelihood | not required | not executed | NOT_APPLICABLE |
| X03 | §15.3 optionalimages/dequantization | not required | not executed | NOT_APPLICABLE |

| OBS01 | §10.1 observed center-to-other-mode mixing ≥20% | separate scientific outcome; prescribed fallback E108 is evaluated independently | E1.json: all four measured fractions below .20 | FAIL |

## Prospective clarifications and prompt/PDF interpretation

- Fixed-budget endpoints: all generative models execute5000updates; final weights are used. ValNLL is monitored and selects Flow depth, not an unrequested training-budget search. No test-based Flow depth choice.
- PDF does not specify the SEM of a difference between models on shared val samples. Use paired per-point NLL difference SEM, ddof1/sqrt5000, declared before E2; also preserve each model's individualSEM.
- RBF MMD² uses empirical V-statistic including diagonal terms, same for all models/noise comparisons. Bandwidth is median of all distinct pairs among a fixed2000train subset,seed604,one pertarget.
- Cov uses exact target mode probabilities (Gaussian disk integrals); Dmode estimates target fractions from the fixed5000test samples, including outliers, consistently in model andfive-noise comparisons. Additionally retain distance to exact target probabilities as an oracle diagnostic. Main E3Q is selected prospectively before E3.
- E1 samples start from fixed standardNormal noise, same random stream acrosseta; center-to-other-mode mixing is a separate5000chain diagnostic from the left center. E3 samples use the same initialization convention. Training replay is separate andnever used to hide poor mixing.
- Quadratic tests may use scipy lfilter to execute the exact finite AR(1) recurrence in float64 with8independent innovation streams; every one of210000steps is simulated and saved, not replaced by stationary analytic draws. Crosscheck its trajectories against the generic autograd sampler.
- If E1 falls back to minimumMMD², PDF defines that metric usingtest; record this exploratory two_blobs selection explicitly. E3 uses a different assigned target, fixedseeds/configs, andno posthoc metric/threshold change.
- PDF explicitly permits recorded energy regularization; initial coefficient is0. No later unregistered regularization or altered scale bounds is allowed.
