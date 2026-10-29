"""Lock one-factor E3 comparison only after all E1/E2 evidence exists."""
import copy,json
from pathlib import Path
from src.utils.io import write_json,digest,file_hash
from reports.tools.lab7_common import R,base,read


def main():
    if (R/'hypothesis.md').exists():raise RuntimeError('Never overwrite a preregistration')
    e1=read(R/'E1.json');e2=read(R/'E2.json');diagnostics=read(R/'diagnostics.json')
    rows=[r for r in e2['rows'] if r['predictor']=='network']
    # Select from observed exploratory configurations by a fixed transparent rule.
    # A is DDPM, B is the smallest effective DDIM K from E2: one sampler configuration factor.
    A={'schedule':'linear','sampler':'ddpm','K':base()['diffusion']['T'],'eta':None}
    B={'schedule':'linear','sampler':'ddim','K':e2['smallest_effective_K'],'eta':0.}
    ra=next(r for r in rows if r['sampler']=='ddpm');rb=next(r for r in rows if r['sampler']=='ddim' and r['K']==B['K'])
    expected=1 if ra['Dmode']>rb['Dmode'] else -1
    lock={'A':A,'B':B,'single_factor':'reverse sampler configuration (DDPM posterior vs deterministic DDIM at lockedK)',
        'fixed_training':'same frozen linear model perseed,20000updates,architecture/data/standardization/optimizer unchanged',
        'primary_metric':'Dmode','difference':'A-B','expected_sign':expected,'seeds':[1,2,3],
        'initial_seed_rule':'7400+seed','DDPM_noise_seed_rule':'7500+seed','samples_per_arm':5000,
        'seed1_reuse':{'A_RUN_ID':ra['RUN_ID'],'B_RUN_ID':rb['RUN_ID'],'training_source':e1['models']['linear']['training_source']},
        'E1_sha256':file_hash(R/'E1.json'),'E2_sha256':file_hash(R/'E2.json'),
        'config':base(),'source_exploratory_commit':'7eb93b2a66e67a1d4135d9a4a3ae8cffb288e04f'}
    write_json(R/'E3_protocol_lock.json',lock)
    text=f'''# Lab7 E3 prospective hypothesis

E1/E2 and diagnostics are complete. This file and E3_protocol_lock.json must be committed in real Git before new E3seed2/3 training/evaluation. Seed1 is explicitly exploratory reuse,not retrospectively labeled unseen.

A={json.dumps(A)}; B={json.dumps(B)}. Sole factor: reverse sampler configuration. All training conditions and the actual frozen linear model are identical within eachseed. DDPM uses posteriorvariance and allTsteps; DDIMeta0 uses the lockedK. Changing the sampling algorithm includes its step-grid by definition (the example explicitly permitted in PDF§9.3),not a second training factor. TargetM6,rho.1,pi0.02; fixedroles701/702/703; train-onlystats; architecturewidth256/depth6/time128/SiLU; Adam.001,batch512,20000updates; no clipping; samebandwidth/reference;5000samples; commonxTseed7400+s; DDPMzseed7500+s. Seeds1/2/3 control initialization,batches,t,epsilon,sampling. No seedselection.

Q=Dmode,Delta_s=Q(A,s)−Q(B,s). Predicted sign={expected:+d}, based on exploratoryDmodeA={ra['Dmode']:.10g},B={rb['Dmode']:.10g}. Eq7.16's stochastic posteriorvariance and Eq7.17's deterministic coarse reverse integration can distribute mode masses differently; Eq7.19 networkerror contributes to both. This is a conditional prediction on this assigned target,not a general guarantee that either sampler is superior.

Prediction is supported only if allthree differences have the expected nonzero sign andabsmean≥2sampleSD, following PDF(7.21). Opposite sign or a teacher decision 'различие не установлено' does not support the main prediction. Metricnoise is independently reported anddifferencesbelownoise are not interpreted as robust performance differences,without replacing the PDFtwo-condition decision.

Seed1 reused exactly: A_RUN_ID={ra['RUN_ID']}; B_RUN_ID={rb['RUN_ID']}; source={e1['models']['linear']['training_source']}. Seeds2/3 use two newlytrained linear models,each shared by A/B. E3 rarityboundaries train two NEW A-targetmodels atpi0/2andpi0/4,seed1,sameAtraining/sampler; do not merely retest oldweights.

Immutable source selections: E1_SHA256={lock['E1_sha256']}; E2_SHA256={lock['E2_sha256']}. Do not revise this hypothesis after E3 results; add separate analysis instead.
'''
    (R/'hypothesis.md').write_text(text,encoding='utf-8')
    print(json.dumps({'A':A,'B':B,'expected_sign':expected},ensure_ascii=False))


if __name__=='__main__':main()
