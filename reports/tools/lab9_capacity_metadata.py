"""Correct descriptive capacity fields, preserving original values and all predictions."""
import yaml
from pathlib import Path
from reports.tools.lab9_common import R,read
from src.utils.io import write_json,file_hash

def main():
    changes=[];corrected={}
    for row in read(R/'run_registry.json'):
        if not row.get('evaluation'):continue
        path=Path(row['path']);cfg=yaml.safe_load((path/'config.resolved.yaml').read_text(encoding='utf-8'));spec=cfg['evaluation_spec'];result=read(path/'results.json');passport=read(path/'manifest.json');raw_hash=file_hash(path/'predictions.pt')
        if spec.get('strategy')=='late':
            capacity=sum(read(Path(spec[k])/'manifest.json')['trainable_parameters'] for k in ['video_path','audio_path'])
            if passport['parameters']!=capacity:changes.append({'RUN_ID':path.name,'field':'manifest.parameters','original_first_constituent_only':passport['parameters'],'correct_composite_total':capacity})
            passport['parameters']=capacity;passport['evaluation_architecture']='Frozen functional late probability mixture at execution; both constituent heads counted. evaluation_spec is authoritative; config.model records representative first constituent.'
            write_json(path/'manifest.json',passport)
        else:capacity=read(Path(spec['path'])/'manifest.json')['trainable_parameters']
        old=result['fusion_profile'].get('parameters')
        if old!=capacity:changes.append({'RUN_ID':path.name,'field':'fusion_profile.parameters','original':old,'correct_trained_head_capacity':capacity})
        result['fusion_profile']['parameters']=capacity;write_json(path/'results.json',result);write_json(path/'summary.json',result);corrected[path.name]=result['fusion_profile']
        assert raw_hash==file_hash(path/'predictions.pt')
    e1=read(R/'E1_exploration.json')
    for r in e1['test'].values():r['fusion_profile']=corrected[r['RUN_ID']]
    write_json(R/'E1_exploration.json',e1)
    e2=read(R/'E2.json')
    for r in e2['models'].values():r['fusion_profile']=corrected[r['RUN_ID']]
    for rs in e2['noise'].values():
        for r in rs:r['fusion_profile']=corrected[r['RUN_ID']]
    for r in e2['supplementary'].values():r['fusion_profile']=corrected[r['RUN_ID']]
    e2['threshold_interpretation']='Highest discrete tested SNR with lower F1 than missing audio, not an interpolated/monotone threshold. Dropout curve is nonmonotone; the30dB difference is small and one-seed descriptive.';write_json(R/'E2.json',e2)
    e3=read(R/'E3.json')
    for r in e3.values():r['fusion_profile']=corrected[r['RUN_ID']]
    write_json(R/'E3.json',e3)
    write_json(R/'execution_failures/capacity_metadata_scope.json',{'changes':changes,'scope':'Descriptive model capacity only: late evaluation passport initially counted representative video constituent instead of both frozen heads. E1 selection/control widths already used correct composite totals before preregistration. No probabilities, metrics, trained weights, model selection, hypotheses or paired statistics changed. Original recorded fields remain in preregistration Git history.','late_public_class_adapter':'Added equivalent public modality-interface adapter after experiments; original functional late outputs retained and unit-checked against the adapter.'})
    print('CAPACITY METADATA',len(changes),'field corrections; raw predictions unchanged')

if __name__=='__main__':main()
