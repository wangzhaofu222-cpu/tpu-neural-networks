# Lab7 requirements: original revision4,16pages, all pages reread

| ID | PDF original requirement | Implementation | Verification | Status |
|---|---|---|---|---|
| A01 | User: verify Lab6 without retraining;153tests,234checks,63checkpoints | separate lab6_verification namespace | hashes,strict loads,regenerated CSV/answers/PNG | PASS |
| A02 | Preserve Lab4Q2 BLOCKED,Lab5split UNVERIFIED,Lab5E2 FAIL,Lab6E1 FAIL | immutable historical records | final hash audit | PASS |
| P01 | §7 exact private-ID lab_pick T/beta,M/rho,pi0 | existing profile generator | independent SHA slices | PASS |
| D01 | §6 unequal circular GMM;20k/5k/5k independent seeds;no cal | gmm2d.py | weighted sampling,roles/hash tests | PASS |
| D02 | §6 train-only coordinate standardization;exactscore same coordinates;metrics original | saved mean/population std/inverse | role isolation and transform tests | PASS |
| S01 | §5.1,(7.4) linear/cosine delta.008 beta≤.999;float64,T+1,recomputeproduct | schedules.py | mandatory group1 | PASS |
| S02 | §8.1 formal alpha_barT≤1e−3;error if violated;explicit shortTdiagnostic | config validation | gate test and E1 boundary record | PASS |
| S03 | (7.2) batched t,2D/imagebroadcast,optionalnoise,one-shot | forward.py | mandatory group2 statistics4SEM | PASS |
| N01 | §8.2 width256,depth6,time128;transformed sinusoidal time everyhidden layer | EpsilonMLP | injection hooks/embedding group6 | PASS |
| N02 | (7.6) mean batch squaredvectornorm;Adam.001,batch512,20000updates | existing ExperimentRunner.run adapter | fullsteps/checkpoints/pairedstreams | PASS |
| N03 | §8.2 parameterization training_target/to_x0/to_epsilon/score | EpsilonParameterization | group3 reconstruction≤1e−8 and erroridentity | PASS |
| N04 | (7.15) exact p/logp/autogradscore/epsilon;unequalweights,standardization,enable_grad | ExactPredictor | finite difference relativeerror≤1e−4,batch stability | PASS |
| R01 | (7.16) DDPM allT,posterior variance,laststepno noise,noclip | samplers.py | group4 NFE/weights/no graph | PASS |
| R02 | (7.17) DDIM roundedtau,K,eta,customvalidtau,finalx0 | samplers.py | group4 eta1full=DDPM≤1e−6;eta0determinism;group5 | PASS |
| R03 | §8.3 fixedxT/z,NFEcallcounter,time/device/n/eta/tau/savedstates | evaluation passports | count and pairednoise checks | PASS |
| G01 | §8.4≥20t incl1/T:Eeps,Ex0,Estar,Enet,fixedval/noise | timestep_error.py | exactEnet0,identity,finiteMCcross-term recorded | PASS |
| G02 | §8.4 41²grid≥4noiselevels;network/exactscore/cosine/lengthratio weightedpt | score_field.py | savedvectors/weights/plots;exactselfcos1 | PASS |
| G03 | §8.4 1000DDPMtrajectories;fractions.8/.6/.4/.2/.05 scaledcenters vsfinalmode | trajectory.py | fullstates/counts/RUN_ID | PASS |
| M01 | (7.20) 5000generated vsfixedtest originalunits;mode<3rho,outlier M | sample_quality.py | independent counts | PASS |
| M02 | §5.6 referencefrequencies100000target;Cov ref/2,Dmodeincloutlier | frozen reference artifact | allmodes inclrare | PASS |
| M03 | §5.6 fixedtrainmedianpairdistance RBFbandwidth | frozen shared bandwidth | no model-specific bandwidth | PASS |
| M04 | §5.6 fivefresh5000target baselines;mean/sampleSD MMD²,Dmode | noise RUNs | recomputation | PASS |
| H01 | §9 E1/E2 hypotheses before exploration,seed1 | exploratory_hypotheses.md | Git chronology | PASS |
| E101 | §9.1 schedules beta/ab/logSNR,terminalKL train,SNR<.01fraction | E1 diagnostics | exact formulas andplots | PASS |
| E102 | §9.1 fixedx0,5t,10000draws eachcoordinate4SEM;clouds entiretrain | forward diagnostic RUNs | all first-draw results retained | PASS |
| E103 | §9.1 two fullmodels onlyschedulechanges,5000DDPMsamples,Lsimple/Eeps/quality | E1 formal RUNs | configdiff and streams | PASS |
| E104 | §9.1 boundaryT/10samebetarange exactDDPM vsfull;terminalKL/quality | diagnostic RUNs | invalid terminal explicitlyflagged,no formal bypass | PASS |
| E201 | §9.2 frozenlinear network DDIM5/10/20/50/100/T+DDPM;sharedxT;5000each | E2 matrix | samples/NFE/config | PASS |
| E202 | §9.2 exactpredictor allsameconditions | exact E2 matrix | no training/mislabeledmodel | PASS |
| E203 | §9.2 warmup+5timings median;K≥10time/NFE within20%ofownmedian | timing records | rawfive times,no cherry picking | PASS |
| E204 | §9.2 exactDDPM/DDIM K=T MMD²≤noisemean+3SD | acceptance | raw quality and fixedthreshold | PASS |
| E205 | §9.2 networkDDIMT threeinitialnoises sMMD,minK difference≤2s | extra evaluation RUNs | savedall3values/sampleSD/rule | PASS |
| E206 | §9.2 boundaryK1/2network/exact;Tweedieposterior mean,samplesenter modes | supplemental RUNs | actualsamples/quality | PASS |
| E301 | §9.3 A/B singlefactor selectedfromE1/E2 | locked protocol | explicit configdiff | PASS |
| E302 | §9.3 hypothesisGit BEFORE E3,sign/fixedvars/mechanism/falsifier | hypothesis.md | ancestor/blob/starttimes | PASS |
| E303 | §9.3 threepairedseeds init/order/t/eps/xT/z;seed1reuseonlyidentical | E3 runs | pairedstream/config/checkpoint evidence | PASS |
| E304 | §9.3 each5000samples,Cov/modefreq/p0/Dmode/MMD² | E3 evaluation | originalrawsamples/reference | PASS |
| E305 | (7.21) threeDeltas,mean,sampleSD,samesign and absmean≥2SD;noisecontext | paired stats | independent arithmetic;no seedselection | PASS |
| E306 | §9.3 boundaryone seed,A,pi0/2 andpi0/4;retrain;trainrarecounts | twofull supplementaltrainings | all20000updates/newdata/trainstats | PASS |
| C01 | §10.1 10T samebeta terminalKL/NFE/time/quality;oneactualtraining | Q1 full20000training | actualDDPM timingandquality | PASS |
| C02 | §10.2 zeropredictor DDPM variance recursion incllastno noise | Q2 sampling | theoreticalvsmeasuredvariance | PASS |
| C03 | §10.3 Lsimple vsquality/SNRfraction/timestep errors | CF answers | actual E1 RUN_IDs | PASS |
| C04 | §10.4 bothEnetT/SNR vsactualDDIM5samples | CF answers | actualvalues/RUN_IDs | PASS |
| C05 | §10.5 maxDmode amongCov1 E3;modes/rare/trajectorytiming | CF answers | no inventedCov1case | PASS |
| T01 | §11 sixmandatorygroups plusresume/uniqueRUN/data/metrics/Git/GPUwhereapplicable | tests/test_lab7_diffusion.py | actualpytest | PASS |
| T02 | User:Lab0–6 regression153tests | originaltests | JUnit153passes | PASS |
| DEL | §12–13 configs/runpassports/weights/RNG/outputs/tables/figures/allCF/finalGit | experiments/lab7,runs/lab7 | independentaudit/regeneration/hashindex | PASS |
| RPT | User explicitly defers formal Russian/LaTeX/PDFreport | no report | scope check | NOT_APPLICABLE |
| X01 | §14.1 optionalx0/vtraining | interfaceextensible,nottrained | optional | NOT_APPLICABLE |
| X02 | §14.2 optionalimages/classifierfreeguidance | notexecuted | optional | NOT_APPLICABLE |
| X03 | §14.3 optionalcrossLab6comparison | notexecuted | optional | NOT_APPLICABLE |

## Interpretation fixed before formal experiments

- PDF(7.20) uses mode frequencies from100000true points for both Cov and Dmode. The prompt's pi0/2 shorthand is reported additionally for raremode; it does not replace the PDF Cov definition. Strict radius<3rho; outlier category M.
- Standardization uses per-coordinate population standard deviation, train only; seeds701/702/703 follow lab-number namespace. Same seeds and latent component uniforms are retained for rare-weight boundary datasets; statistics are recomputed from their own train.
- Lsimple is mean over samples of SUM over2coordinates. Eq7.18 diagnostic errors additionally divide by dimension2. Neither is NLL.
- Eq7.19 holds in population. Record the finite-val cross term and its SEM; do not force sample estimates to add exactly or assume sample noise can never favor network.
- RBF MMD² uses the same empirical V-statistic including diagonals as Lab6. Bandwidth is median of all distinct train pairs (all20000trainpoints, exact blocked distances); one fixed original-unit value throughoutLab7. E3 boundary targets retain this main bandwidth for comparison and have their own100000point mode references/noise baselines.
- SiLU is used in all hidden/time layers; PDF specifies dimensions and injection but no activation. Time is injected before each hidden activation. The architecture/device is locked after actual budget testing.
- All primary formal endpoints are20000updates, val is monitored without selecting an extra training budget. E3 may reuse exact E1seed1 conditions with explicit provenance.
- The intentional T/10 terminal violation is diagnostic-only. 10T Q1 is actual full-budget training, not a schedule calculation alone.
- No condition is reseeded to obtain a passing stochastic threshold. Timing failures and scientific threshold failures remain visible.
