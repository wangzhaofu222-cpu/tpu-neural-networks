# Lab 1: complete source-derived execution plan

Source: lab1_trainability_1010.pdf, revision 4, 2026-10-08, all 13 pages reread as text and rendered pages. Sections 14 (extra experiments) and 15 (oral control questions) are separate from mandatory deliverables.

## Fixed protocol (written before formal runs)

- Profile P3adf02: depths 8,24,48,96; bad Gaussian sigma=.02; SGD momentum=.9. Hash computation verified without persisting the original identifier.
- CIFAR-10 is explicitly permitted (§6). Reuse Lab0 data/split/normalization: full training role 43000, validation 5000, calibration 2000, test 10000. No test access for selection. All formal Lab1 training uses all 43000 training examples, not Lab0's 25% subset.
- Width256, ReLU, no normalization, biases zero, batch128, float32 training, no AMP, 3 full epochs, weight decay0, constant LR selected once from .001,.003,.01,.03,.1 on shallow plain MLP using final val accuracy, then val loss, then smaller LR. No gradient clipping (would obscure failure). Existing ExperimentRunner, checkpoints, provenance, data and RNG retained.
- Probe: first64 examples of fixed training subset, eval mode, unit vector from profile base seed. Float64 copy of identical float32 weights and float64 inputs, preventing numerical underflow in deep Gaussian probes. Training model/buffers/RNG/gradients unchanged. Scope and precision recorded for every run. rho_T evaluates final weights on first1024 fixed training examples, not moving epoch loss.
- GI includes all directly parameterized layers including input/head, uses median per-layer g over first/middle/last minibatch each epoch, then max/min with epsilon1e-12. AVD on fixed eval probe hidden outputs (residual includes input projection h0); raw per-layer arrays retained. ReLU zero fractions apply to ReLU outputs; residual block outputs are signed sums (their zero fraction is observational, not a dead-ReLU claim).
- Residual depth L counts blocks with 2L linear layers plus common input projection and classifier. Plain depth L counts L hidden linear layers plus classifier. Exact parameters are recorded. Branch has no post-add activation.
- Expected E1: chi He1, Xavier.5, Gaussian.0512 (§5 equations1.4–1.8); gradient-output and parameter-gradient attenuation are distinct. E2: unit residual q grows approximately3^L; scaled residual bounded by exp(c), c measured first branch (§5.5). Training may diverge; retain numerical failure as real result.

## Mandatory completion gates

| Gate | Required output / acceptance |
|---|---|
| §8.1 monitor | mean,std,var,q,min,max,ReLU zeros,tanh derivative<.1; gradient norm,g,relative norm,delta squared mean; observational hooks; float64 statistics |
| §8.2 model | plain/residual, configurable depth,width,ReLU/tanh,none/BN/LN before activation, explicit He/Xavier/Gaussian, alpha1/inverse-sqrt L; fixed real probe B>=64 |
| §8.3 summary | rho_T, GI, AVD, max zero/saturation, numerical flag; summary.json and compare table |
| E1 | 3 inits x4 depths, optimizer-free; log-layer q/delta/g plots; geometric chi; exclude q<1e-30 with effective depth; max-depth fwd/bwd relative error<=20% or explicit localized diagnosis; Gaussian repeated after one full training epoch |
| E2 | 5-LR shallow selection; 3 architectures x4 depths x3 full epochs; before-training qL/q0,c; unit growth relative error<=20%; scaled qratio<=1.2 exp(c) at all depths; rho/GI/AVD/val and first untrainable depth (rho>=.9 or numerical failure), explicitly none if absent |
| Map | depth x architecture -> rho_T, fixed factors explicitly listed; failed numeric cells marked as failures, never substituted numeric results |
| E3 | choose two models from E1/E2 differing exactly one factor; actual Git hypothesis before all E3; same literal seeds1,2,3; all paired deltas, sample std(ddof1), same sign and abs(mean)>=2std; if not met state difference not established; limitations one fixed split |
| CF1 | measured He chi predicts q256/q1=chi^255, finite width/correlation limitations and RUN_ID |
| CF2 | width1024 same sigma predicts chi.2048; additional E1 RUN_ID |
| CF3 | observed delta attenuation vs near-constant internal g, explain product/cancellation, avoid treating epsilon-floor GI as proof |
| CF4 | own measured growth for all unit residual depths vs (1+c)^L, explain instability despite identity |
| CF5 | select two E2 configs at depth<96 with close val accuracy and materially different GI; examine next profile depth; explicitly exploratory selection; additional run if existing data insufficient |
| Tests §11 | unchanged outputs; unchanged gradients; all init variances within10% for fan product>=1e4; zero residual branch identity<=1e-6; NaN/Inf numerical failure flag |
| Report §12 | six Russian LaTeX sections: profile/protocol; E1; E2; map; E3; five CF. Auto-derived numbers, tables and figures tied to RUN_IDs. Actual compilation and every-page visual check |
| Final | repeat regression tests, source PDF audit, Lab0 artifact hashes unchanged, evidence gate by gate, final real Git commit |

Small/synthetic tests and budget runs are labelled separately and cannot satisfy formal experimental gates. E3 hypothesis will be chosen only after E1/E2 and committed before its six runs.
