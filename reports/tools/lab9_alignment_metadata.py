"""Keep canonical time windows even when a face is absent; never alter pooled inputs."""
from pathlib import Path
import shutil,torch
from reports.tools.lab9_common import R,D,FEATURE,read
from src.multimodal.alignment import alignment_windows
from src.utils.io import file_hash,write_json

def complete_alignment(c,v,a):
    n=torch.load(c['video_frames_path'],weights_only=True)['canonical_video_count'];indices=v['indices'];aligned=alignment_windows(range(n),len(a['z']),av_offset_ms=c['av_offset_ms']);valid=torch.tensor([k in indices for k in range(n)])
    recorded={f['raw10fps_index']:f for f in c.get('frames',[])}
    reasons={str(k):('face_detector_missing' if 2*k in recorded and not recorded[2*k]['face_found'] else 'no_decoded_frame_at_canonical_slot') for k in range(n) if k not in indices}
    return {'clip_id':c['clip_id'],'canonical_video_indices':list(range(n)),'visual_original_indices':indices,'visual_valid':valid,'visual_missing_reasons':reasons,'audio_valid':torch.ones(len(a['z']),dtype=torch.bool),'encoded_visual_windows':[aligned['windows'][k] for k in indices],**aligned}

def main():
    before=file_hash(FEATURE);affected=[];total=0
    for c in read(R/'data_manifest.json')['clips']:
        if c['role']=='cal':continue
        v=torch.load(D/'features/visual'/(c['clip_id']+'.pt'),weights_only=False);a=torch.load(D/'features/audio'/(c['clip_id']+'.pt'),weights_only=False);p=D/'features'/(c['clip_id']+'_alignment.pt');old=torch.load(p,weights_only=False);new=complete_alignment(c,v,a)
        if not new['visual_valid'].all():
            backup=R/'execution_failures/alignment_before_canonical_windows';backup.mkdir(exist_ok=True)
            if not (backup/p.name).exists():shutil.copy2(p,backup/p.name)
            original=torch.load(backup/p.name,weights_only=False)
            affected.append({'clip_id':c['clip_id'],'missing_canonical_indices':torch.where(~new['visual_valid'])[0].tolist(),'missing_reasons':new['visual_missing_reasons'],'old_assigned_audio_count':len(original['assigned_indices']),'new_assigned_audio_count':len(new['assigned_indices']),'old_copy':str(backup/p.name),'old_sha256':file_hash(backup/p.name)})
        # Audio outside the finite video support is explicitly outside coverage, not an internal gap.
        times=new['audio_sample_times'];n=len(new['canonical_video_indices']);inside=(2*times>=-3200)&(2*(times-(n-1)*3200)<3200);assigned=new['assigned_indices'];assert set(assigned.tolist())==set(torch.where(inside)[0].tolist())
        assert len(assigned)==len(torch.unique(assigned));torch.save(new,p);total+=1
    assert before==file_hash(FEATURE)
    from collections import Counter
    reasons=Counter(reason for row in affected for reason in row['missing_reasons'].values())
    write_json(R/'alignment_metadata_audit.json',{'status':'PASS','clips':total,'affected_clips':affected,'missing_slot_reasons':dict(reasons),'pooled_features_sha256_before':before,'pooled_features_sha256_after':file_hash(FEATURE),'scope':'Canonical windows include all original slots, with explicit false visual validity. Three face-detection failures and terminal canonical slots without a decoded frame are distinguished; no synthetic/repeated frame was inserted. Previously only encoded face slots were exported. Pooled means, learned weights, predictions and experiment conditions are unchanged. Outside finite video support is outside coverage, not an internal missing window.','cal_read':False})
    print('CANONICAL ALIGNMENT',total,'affected',len(affected),'pooled inputs unchanged')

if __name__=='__main__':main()
