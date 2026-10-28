# Lab4 requirements — original PDF reread, revision4 (2026-10-08)

Personalprofile:tau=.1,thirdop=flip,fractions=.02/.05/.25/1,rmin=.02. ExactstudentID requested byuser is retained only in ignored .private/lab4_student_id.json; publicfiles keephash.

PrimaryGPUprotocol:N256,Efull100,Eshort25. NoCPUsmallprotocolsubstitution. CF1 uses Eshort25 atN64 (not anindependent-smallbatch approximation). Probe usescachedfeatures andsameExperimentRunner. Cal access forbidden. Q2 remainsBLOCKED becauseProtocol§4.6 ismissing; allotherworkproceeds.

| ID | Originalsource | Mandatory/optional | Requirement | Implementation | Verification | RUN_ID | Status |
|---|---|---|---|---|---|---|---|
|A01|§1,§8.2|mandatory|Reuse Lab0 training/provenance/RNG/checkpoints; no new training framework|src/engine/runner.py;shared factories|old88 regression +SSL resume tests|—|PENDING|
|A02|§7|mandatory|SHA256 student_id:lab4 slices0:2/2:4/4:6 ->tau.1,flip,[.02,.05,.25,1]|src/profile_gen.py;configs/lab4.yaml|profile.json;rawID only .private/lab4_student_id.json|—|PENDING|
|D01|§6|mandatory|Immutable Lab0 CIFAR43000/5000/2000/10000 roles|src/data/ssl_data.py|split_fingerprint,actualindices,role access tests|—|PENDING|
|D02|§6|mandatory|SSL reads onlytrainimages, no targets; no val/test selection or earlystop; cal blocked|src/data/ssl_data.py|target-access spy and source audit;monitor trainonly|—|PENDING|
|V01|(4.1)-(4.2),§8.1|mandatory|Independent two views; fiveops configurable; stablehash;Tminus generated;noaugment|src/ssl/views.py;augmentations.py|view0/2workers tests and8imagegrid|—|PENDING|
|V02|§8.1 example|mandatory|RRCp1 scale.08..1;jitterp.8 strength.4/.4/.4/.1;flipp.5 (profile)|configs/lab4.yaml|savedresolvedconfig andaug_hash|—|PENDING|
|M01|(4.3)|mandatory|CIFARResNet18 3x3,noinitialmaxpool,h512;projector512-ReLU-128 nobias|src/ssl/simclr.py|state_shapes andforward/backward|—|PENDING|
|L01|(4.4)-(4.5),§8.2|mandatory|Concatenate2N,normalize,positive(i+N)%2N,diag-inf,logsumexp/tau,mean|src/ssl/losses.py|knownsame/orthogonal/tau.01 cases1e-5|—|PENDING|
|L02|§8.2|mandatory|Return pos/negsim,N,2N-2,ln(2N-1);noexistingSimCLRloss|src/ssl/losses.py;runner|step logs andmanifest|—|PENDING|
|L03|(4.6),§11.3|mandatory|Anchor gradient onalready-normalizedz equalsformula within1e-5|tests/test_lab4_ssl.py|autograd isolatedanchor term|—|PENDING|
|F01|§8.3|mandatory|Extracth/zhat train,val,test eval/noaugment/no_grad/trainnormalization;cacheparentRUN|src/evaluation/features.py|fullstate hash unchanged;repeat error<=1e-6|—|PENDING|
|P01|(4.10),§8.3|mandatory|Frozenfeaturelinearprobe;regularizationvalonly;testfinal;commonmethodallencoders|src/evaluation/linear_probe.py;sharedrunner|full encoderstatecheck,configmatrix|—|PENDING|
|K01|(4.11)|mandatory|WeightedcosinekNN trainonly,k200,tau.07|src/evaluation/knn.py|roleguardandknownvectors|—|PENDING|
|R01|(4.7)-(4.8)|mandatory|5000fixedtrainids;norm,trace,eigenspectrum,reff,reff/d|src/diagnostics/representation.py;ssl_monitor.py|savedrawfeatures/spectrum anddiagnosticids|—|PENDING|
|R02|§5.5|mandatory|trace<=1e-6*d*mean_squared_norm =>reffnull,collapse_scale;dimensionalratio<.2|representation.py|identical/rank1/Gaussian20d tests|—|PENDING|
|R03|(4.9)|mandatory|zhatmeanvecnorm,traceidentity;pospairandrandompairsims fixedseed|ssl_monitor.py|identityerror<=1e-5;perexample rawsimilarities|—|PENDING|
|H01|§9|mandatory|E1/E2 exploratory hypotheses before experiments;theorybasis|experiments/lab4/exploratory_hypotheses.md|realGitcommit before realruns|—|PENDING|
|E101|§9 common|mandatory|GPU N256,Efull100,Eshort25,tauprofile;commonSSLoptimizer/schedule|configs/lab4.yaml|formalconfigguards,resolvedconfigs|—|PENDING|
|E102|§9.1|mandatory|B0randomno training;B1fulltrainlabels25epochcropflip;SSLfullT100|sharedrunner run(initialize_only=True) andrun()|distinctRUN_IDs andexactepochlogs|—|PENDING|
|E103|§9.1|mandatory|EverySSLepoch loss,pos/negsim,trace/rankh/zhat,meanvecnorm,collapse|ssl_monitor.py;runner|epoch0..100 rawJSON/PT;NT-Xentstepmetrics|—|PENDING|
|E104|§9.1|mandatory|Probe>=5SSLcheckpoints including0;initial/middle/finaleigenspectra|checkpoints0/25/50/75/100 +evaluationpipeline|5actualprobe evaluations/cachedfeatures|—|PENDING|
|E105|§9.1|mandatory|B0/B1/SSL sameprobe/kNNtest andsummary;loss<=ln511;G1sign|evaluationpipeline|results.json gates;negativeG preserved|—|PENDING|
|E106|§9.1 boundary|mandatory|Findlossdecline/probeplateau phase orstateabsent;lossnotsolequality|savedmetricsanalysis|explicitcomputedinterval rule,results plot|—|PENDING|
|E201|§9.2|mandatory|Six25epochseed1 SSLvariants:T,minus each3,onlylargestdeltaop,none|programmatic augmentation variants|realRUNs;T25neverreplacedbyT100|—|PENDING|
|E202|§9.2|mandatory|AllE2sameN,tau,network,opt/schedule,epochs;onlyaugschange|src.compare.diff_configs|savevariable-isolation report|—|PENDING|
|E203|§9.2|mandatory|EachE2probe/kNN/rank/trace/collapse;deltaM(T)-M(T-j),G0|savedencoder summaries|barplots,rawcomparison|—|PENDING|
|E204|§9.2 thresholds|mandatory|critical:G0>0 anddelta>=G0/2 ORcollapse;harmful:nocollapse,G0>0,delta<=-G0/2;otherwise noise;G0<=0 nohalfgainrule|src/analysis/label_efficiency.py|classificationtests andrawgates|—|PENDING|
|E205|§9.2 boundary|mandatory|Nameinvariance;onlymostimportantop andnone;no-collapseallowedexplainedvia4.6|augmentation variants andinterpretation|actualsingle/noaugresults;single-seedlimitation|—|PENDING|
|E301|§9.3|mandatory|Noencoderretraining:E1fixedB0/B1/SSL;A=SSL,B=B0|featurecaches|checkpointshaunchanged|—|PENDING|
|E302|§9.3|mandatory|BeforeE3hypothesisGit:A/B,rmin,subsets,probe,sign+magnitude,randomsources,imageunit,falsification|experiments/lab4/hypothesis.md|Gitancestry andtimestamps before newE3probes|—|PENDING|
|E303|§9.3|mandatory|36probes:3encoders x3seeds x4fractions;stratifiednestedpairedids;r1alltrain|label_subsets;cachedfeatures;sharedrunner|savedlabelindices/manifests and36matrix|—|PENDING|
|E304|(4.12)-(4.13)|mandatory|3pairedrmin deltas,mean,ddof1SD;nonzero samesign andabsmean>=2SD else unestablished|label_efficiency.py|actualstatistics;unchangedteacherthreshold|—|PENDING|
|E305|§9.3|mandatory|M(r),G(r),all3seedpoints,logr;seedsnotpretraining;val5000optimisticfew-labelcondition|results plotting|plots andexplicitlimitations|—|PENDING|
|C01|§10Q1|mandatory|First/lastE1loss vsln511;actualN64shortcounterfactual|CF1 N64 Eshort25,allotherconditionssame|stepsandln127/neg126 recorded|—|PENDING|
|C02|§10Q2|mandatory|Hue-definedcontrol perteacherProtocol§4.6;probeE2with/withoutjitter|missing_resources.json|NoProtocolfound;cannotinventdefinition|—|BLOCKED|
|C03|§10Q3|mandatory|Actualpair similarreff differentprobe ORreverse;scaleconvention|E1/E2summaryanalysis|closestactualpairs/numbers/RUN_IDs|—|PENDING|
|C04|§10Q4|mandatory|FinalSSLprobe h vszhat actual;headnotautomaticallyharmful|sharedprobe feature_kind|newzhatprobeRUN andpairedhresult|—|PENDING|
|C05|§10Q5|mandatory|CompareG(rmin)/G1;B1fairlabelbudget extraactualrun|CF5 supervisedtrainrmin25epochs,seed1,otherwiseB1same|labelsubsetmanifest,conditiondiff,actualprobe|—|PENDING|
|T01|§11.1|mandatory|Views shape/samewhenoff/differentwhenon/workers>0|tests/test_lab4_ssl.py|actualpytest|—|PENDING|
|T02|§11.2|mandatory|NT-Xentknownloss andtau.01finite tol1e-5|tests/test_lab4_ssl.py|actualpytest|—|PENDING|
|T03|§11.3|mandatory|Anchorgradient eq4.6 tol1e-5|tests/test_lab4_ssl.py|actualpytest|—|PENDING|
|T04|§11.4|mandatory|Probe fullencoderstate/BNbuffers unchanged;kNN trainonly|tests/test_lab4_ssl.py|actualpytest|—|PENDING|
|T05|§11.5|mandatory|Rank1=1;Gaussiann20d rank>=.9d;identicalnull/collapse;eq4.9 tol1e-5|tests/test_lab4_ssl.py|actualpytest|—|PENDING|
|T06|§11.6|mandatory|Nestedstratifiedsubsets/calaccesserror|tests/test_lab4_ssl.py|actualpytest|—|PENDING|
|T07|Userengineering|mandatory|Resume/RNG/featuresrepeat/workers/pairing/RUNunique/stats/leakage/old88regression|tests/test_lab4_ssl.py +oldsuite|actualJUnit|—|PENDING|
|O01|§12-13/user|mandatory|Allconfigs,rawRUNlogs,localckpt/features/hashindices,summaries,plots,tests,finalGit|runs/lab4;experiments/lab4|finalaudit;nohistoricaloverwrite|—|PENDING|
|O02|Useroverride§12|mandatory|NoLaTex/PDFformalreport;stopbeforeLab5|scope|NOT_APPLICABLEreport|—|NOT_APPLICABLE|
|X01|§14.1|optional|Temperaturegrid5values andSGD-taucontrol|notrequested|notrun|—|NOT_APPLICABLE|
|X02|§14.2|optional|Negative-freecollapse stresstest|notrequested|notrun|—|NOT_APPLICABLE|
|X03|§14.3|optional|Facetransfer/RAVDESSactordata|notrequested;dedicatedpackmissing|notrun|—|NOT_APPLICABLE|

ImplementationchoicesnotfixedbyPDF: SGDmomentum.9,SSLlr.3,wd1e-4,10epochwarmup+cosine,horizon100/25;samepolicyallSSL. Directsimultaneous2Nforward preservesBatchNorm andglobalNT-Xent. AMPwillbeusedonlyafterN256memory/numericaltests; losscomputedfloat32. Linearprobe fullbatchSGDmom.9,lr.1,200epochs,train-onlyfeaturestandardization,L2grid0/.001/.01 chosenbyval;testonlyfinal. Initial/mid/finalencodercheckpoints are0/25/50/75/100. E2critical/harmfulclassificationis exploratory singleseed.
