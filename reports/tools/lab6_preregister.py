"""Create prospective E3 configuration lock after E1/E2, never edit it afterward."""
from pathlib import Path
from datetime import datetime,timezone
from src.generative.experiments import config,read
from src.utils.io import write_json,digest,file_hash

R=Path('experiments/lab6')
def main():
    if (R/'hypothesis.md').exists() or (R/'E3_protocol_lock.json').exists():raise RuntimeError('Prospective E3 record already exists; refusing to overwrite')
    e1=read(R/'E1.json');e2=read(R/'E2.json');match=read(R/'E3_capacity_match.json');profile=read(R/'profile.json')['profile']
    if match['relative_gap_over_smaller']>.2:raise RuntimeError('Capacity mismatch')
    eta=e1['selected_eta'];depth=e2['selection']['selected_depth'];configs={}
    for seed in [1,2,3]:
        for kind in ['ebm','flow']:
            configs[f'{kind}_{seed}']=config(kind=kind,target=profile['lab6_target'],seed=seed,role='E3_'+kind,
                width=match['ebm_width'] if kind=='ebm' else 64,depth=depth,eta=eta)
    lock={'recorded_at':datetime.now(timezone.utc).isoformat(),'target':profile['lab6_target'],'eta':eta,'K':50,'temperature':1.,
        'depth':depth,'cap':profile['lab6_scale'],'capacity_match':match,'primary_metric':'MMD2','A':'EBM','B':'Flow',
        'difference':'MMD2(EBM)-MMD2(Flow)','expected_sign':'positive','seeds':[1,2,3],
        'config_hashes':{k:digest(v) for k,v in configs.items()},'resolved_configs':configs,
        'E1_selection_sha256':file_hash(R/'E1.json'),'E2_selection_sha256':file_hash(R/'E2.json')}
    write_json(R/'E3_protocol_lock.json',lock)
    text=f'''# Lab6 E3 prospective primary comparison

This immutable hypothesis and its resolved configuration lock must be committed to actual Git BEFORE any E3 model training. It is created after E1 and E2 completed, not reconstructed from E3 outcomes. Future interpretation belongs in a separate analysis file.

Q = MMD² (saved field MMD2), empirical RBF V-statistic including diagonals.
A = EBM; B = Flow.
Delta_s = Q(EBM,s) - Q(Flow,s).
Expected sign = positive (Flow has lower sample-distribution discrepancy).

## Fixed protocol

Target = {profile['lab6_target']}, original unstandardized coordinates, Omega=[-15,15]². Train20000seed601,val5000seed602,test5000seed603; no cal. All six models execute5000Adam updates,lr.001,batch256,constant scheduler,float32,CPU2threads,seeds1/2/3. All paired arms share dataset,mini-batch order by a private loader generator,optimizer-step budget,evaluation code and5000generation samples. Each pair's generation uses seed7000+seed for standardNormal initial noise; EBM continues with its Langevin noise, Flow applies its inverse. Fixed-budget endpoint weights are evaluated. No checkpoint/test-driven tuning.

EBM keeps three hidden ReLU layers, external s3 quadratic, bounded tanh residual, replay10000,restart.05,K50,T1,eta={eta} selected by E1. Energy regularization coefficient0. E1 mixing criterion satisfied={e1['mixing_rule_satisfied']}; selection rule: {e1['selection_rule']}. The inherited step is not assumed to mix all eight modes.

Flow keeps E2-selected Kf={depth},Kf−1swaps,c={profile['lab6_scale']},two separate s/t networks perblock,each2hidden64ReLU. Its exact testNLL remains distinct from EBM grid estimates.

The permitted capacity adjustment changes only EBM hidden width from128 to{match['ebm_width']}. EBM parameters={match['ebm_parameters']};Flow parameters={match['flow_parameters']};gap over the smaller count={match['relative_gap_over_smaller']:.8f},≤20%. No other depth,Kf,eta,K,temperature,target or optimizer-budget change is allowed. Exact six configurations and their hashes are in E3_protocol_lock.json.

## Prediction and falsification

Expect positive meanDelta: a bounded residual can change energy by at most2 while the s3 Gaussian prior is broad relative to sigma.2 target modes. Short-run K50 Langevin may retain initial/noise/replay-related biases. A trained flow fits exact density by(6.9) and samples by explicit inversion, so we expect smaller MMD². This is a model/protocol prediction, not proof that likelihood universally implies sample quality or that an observed gap has only one cause.

The direction is contradicted by nonpositive meanDelta. Inconsistent signs or failure of the teacher statistical/noise gates leave the difference unestablished even if the mean is positive. Declare 'различие установлено' only if all three nonzero differences have one sign,abs(mean)>=2*sampleSD(ddof1),AND abs(mean)>=MMD² noise mean from five fresh5000target-vs-fixedtest comparisons. Otherwise declare 'различие не установлено'. Show everyDelta,mean,SD andnoise. Do not replaceQ,reverseA/B,selectseeds,changebandwidth or tune width after observing outcomes.

The bandwidth is determined once from fixed2000trainpoints,seed604. The noise baseline follows seeds6101..6105, with all raw samples saved. Dmode andCov are secondary descriptors, not substitutes for the preregistered primary test.

## Midpoint diagnostic and scope

Nearest-center pair is chosen solely by geometricdistance,rounded tie-breaking by original center indices. Record true logdensity,Flow exact logdensity andEBM Omega-grid estimated logdensity at its midpoint,plus the full201pointline and densitymaps,for allthree seeds. Report excess logdensity continuously; do not presume a visible bridge or attribute every gap solely to invertibility. Positive density everywhere is not itself evidence of a large visible bridge. No further model training/tuning follows E3 comparison. Noformalreport orLab7.
'''
    (R/'hypothesis.md').write_text(text,encoding='utf-8');print({'eta':eta,'depth':depth,'capacity':match,'primary':'MMD2 EBM minus Flow'})
if __name__=='__main__':main()
