"""Descriptive raw/preprocessed data audit and face contact sheet; excludes cal."""
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
from reports.tools.lab9_common import R,read
from src.utils.io import write_json

def main():
    clips=[c for c in read(R/'data_manifest.json')['clips'] if c['role']!='cal'];frames=[f for c in clips for f in c['frames']];selected=[f for f in frames if f['selected5fps']]
    byactor={}
    for c in clips:byactor.setdefault(c['actor_id'],c)
    out=R/'data_QA';out.mkdir(exist_ok=True);canvas=Image.new('RGB',(6*224,4*250),'white');draw=ImageDraw.Draw(canvas);sampled=[]
    for k,(actor,c) in enumerate(sorted(byactor.items())):
        f=next(f for f in c['frames'] if f['face_found']);p=Path('data/lab9/preprocessed')/c['clip_id']/f"frame_{f['raw10fps_index']:03d}.jpg";im=Image.open(p);assert im.size==(224,224)
        canvas.paste(im,((k%6)*224,(k//6)*250));draw.text(((k%6)*224+5,(k//6)*250+227),f"Actor {actor:02d} / {c['role']}",fill='black');sampled.append({'actor_id':actor,'clip_id':c['clip_id'],'path':str(p),'role':c['role']})
    canvas.save(out/'face_grid.png')
    data={'noncal_clips':len(clips),'actors':len(byactor),'raw10fps_frames':len(frames),'face_not_found_raw10fps':sum(not f['face_found'] for f in frames),'selected5fps_slots':len(selected),'face_not_found_5fps':sum(not f['face_found'] for f in selected),'clips_entirely_missing_video':sum(c['valid5fps_frames']==0 for c in clips),'audio_samples_minmax':[min(c['audio_samples'] for c in clips),max(c['audio_samples'] for c in clips)],'mel_time_steps_minmax':[min(c['audio_mel_shape'][1] for c in clips),max(c['audio_mel_shape'][1] for c in clips)],'offset_ms_counts':dict(Counter(str(c['av_offset_ms']) for c in clips)),'header_decoder_difference_counts':dict(Counter(str(c['decoder_header_frame_difference']) for c in clips)),'face_grid_samples':sampled,'cal_media_read':False,'QA_status':'Contact sheet generated; visual inspection recorded separately'}
    write_json(R/'data_quality.json',data);print({k:v for k,v in data.items() if k!='face_grid_samples'})

if __name__=='__main__':main()
