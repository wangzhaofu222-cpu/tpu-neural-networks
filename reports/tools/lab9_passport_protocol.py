"""Mirror verified immutable input identities into every Lab9 passport, with chronology."""
from pathlib import Path
from datetime import datetime,timezone
from reports.tools.lab9_common import R,FEATURE,read
from src.utils.io import write_json,file_hash

def main():
    source=read(R/'encoder_sources.json');manifest=read(R/'data_manifest.json');offsets={c['av_offset_ms'] for c in manifest['clips'] if c['role']!='cal'};assert offsets=={0.}
    common={'eta_v_fps':5,'raw_face_fps':10,'fs':16000,'n_hop':160,'r':1,'tau_w_seconds':.2,'window_samples':3200,'av_offset_ms':0.,'offset_source':'per-clip measured PTS in data_manifest.json; all1320 are0ms','window_boundary':'left-inclusive/right-exclusive, integer samples','cal_media_read':False,'mel_normalization_sha256':file_hash(R/'mel_normalization.json'),'data_manifest_sha256':file_hash(R/'data_manifest.json'),'preprocessing_protocol_sha256':file_hash(R/'media_preprocessing_protocol.json'),'annotation_timing':'Mirrored after execution from immutable source/cache protocol; not represented as newly preregistered information','annotation_utc':datetime.now(timezone.utc).isoformat()}
    count=0
    for row in read(R/'run_registry.json'):
        p=Path(row['path'])/'manifest.json';passport=read(p);protocol=dict(common)
        if row['kind']=='AudioCNN_train_val_selected':protocol.update(phase='audio_encoder_training',frozen_encoder_inputs=None,produced_audio_checkpoint_sha256=source['audio']['sha256'],input='train-only standardized logmel; val-only stopping/selection')
        else:protocol.update(phase='frozen_feature_head_or_evaluation',frozen_visual_checkpoint_sha256=source['visual']['sha256'],frozen_audio_checkpoint_sha256=source['audio']['sha256'],clip_feature_normalization_sha256=file_hash(R/'feature_normalization.json'),immutable_pooled_feature_sha256=file_hash(FEATURE))
        passport['multimodal_protocol']=protocol;write_json(p,passport);count+=1
    write_json(R/'passport_protocol_annotation.json',{'passports':count,'scope':'Explicit timing/alignment/encoder hash metadata added from the saved input protocol; original configs, config hashes, input fingerprints, source commits, weights, predictions and statistics unchanged. Audio training is distinguished from later frozen-audio use. Original passports remain in Git history.','immutable_pooled_feature_sha256':file_hash(FEATURE)});print('PASSPORT PROTOCOL',count)

if __name__=='__main__':main()
