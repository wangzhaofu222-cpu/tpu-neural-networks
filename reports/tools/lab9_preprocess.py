"""Stream real official AV clips to 10FPS face frames, selected5FPS and16kHz audio."""
import hashlib,json,math,re,subprocess,time,tempfile
from pathlib import Path
import cv2,numpy as np,torch
from scipy.io import wavfile
import imageio_ffmpeg
from src.multimodal.audio import log_mel
from src.utils.io import write_json,file_hash

R=Path('experiments/lab9');ROOT=Path('data/lab9');OUT=ROOT/'preprocessed'

def process(clip,detector,ffmpeg):
    cid=clip['clip_id'];dest=OUT/cid
    if (dest/'metadata.json').exists():return json.loads((dest/'metadata.json').read_text())
    assert clip['role']!='cal';dest.mkdir(parents=True,exist_ok=True);source=Path(clip['media_path']);wave=dest/'audio.wav'
    result=subprocess.run([ffmpeg,'-hide_banner','-nostdin','-y','-i',str(source),'-map','0:a:0','-ac','1','-ar','16000','-af','ashowinfo','-c:a','pcm_f32le',str(wave)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    (dest/'decode.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Audio decode failed '+cid)
    fs,audio=wavfile.read(wave);assert fs==16000 and audio.ndim==1 and np.isfinite(audio).all()
    mel=log_mel(audio);torch.save(mel,dest/'log_mel.pt')
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert fps>0 and n>0
    duration=n/fps;count=math.ceil(duration*10-1e-8);targets={min(n-1,int(math.floor(j*fps/10+.5))):j for j in range(count)};frames=[];indices=[];metadata=[];video_start=None
    decoded=0
    for i in range(n+2):
        ok,frame=cap.read()
        if not ok:break
        decoded+=1
        if i==0:video_start=float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
        if i not in targets:continue
        j=targets[i];h,w=frame.shape[:2];scale=320/w;small=cv2.resize(frame,(320,max(1,round(h*scale))));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(24,24));found=len(faces)>0;box=None
        if found:
            xx,yy,ww,hh=max(faces,key=lambda z:int(z[2])*int(z[3]));pad=.15
            x0=max(0,round((xx-pad*ww)/scale));y0=max(0,round((yy-pad*hh)/scale));x1=min(w,round((xx+(1+pad)*ww)/scale));y1=min(h,round((yy+(1+pad)*hh)/scale));box=[x0,y0,x1,y1]
            face=cv2.resize(frame[y0:y1,x0:x1],(224,224),interpolation=cv2.INTER_AREA)
            ok_jpg,jpg=cv2.imencode('.jpg',face,[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok_jpg:raise RuntimeError('JPEG encoding failed')
            (dest/f'frame_{j:03d}.jpg').write_bytes(jpg.tobytes())
            if j%2==0:frames.append(torch.from_numpy(cv2.cvtColor(face,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1));indices.append(j//2)
        metadata.append({'clip_id':cid,'actor_id':clip['actor_id'],'role':clip['role'],'raw10fps_index':j,'source_frame_index':i,'source_time_s':i/fps,'canonical_time_s':j/10,'face_found':bool(found),'box_source_pixels':box,'selected5fps':j%2==0})
    cap.release()
    if abs(decoded-n)>1:raise RuntimeError(f'Unexpected decoder/header frame count discrepancy {decoded}/{n}: '+cid)
    duration=decoded/fps
    torch.save({'frames':torch.stack(frames) if frames else torch.empty(0,3,224,224,dtype=torch.uint8),'video_indices_5fps':indices,'canonical_video_count':math.ceil(duration*5-1e-8)},dest/'video_frames.pt')
    match=re.search(r'Parsed_ashowinfo[^\n]*pts_time:([-+.0-9e]+)',result.stderr);audio_start=float(match.group(1)) if match else 0.
    meta={**clip,'status':'PASS','audio_path':str(wave),'video_frames_path':str(dest/'video_frames.pt'),'log_mel_path':str(dest/'log_mel.pt'),'source_sha256':file_hash(source),'audio_sha256':file_hash(wave),'video_frames_sha256':file_hash(dest/'video_frames.pt'),
          'sample_rate':16000,'audio_samples':len(audio),'audio_mel_shape':list(mel.shape),'source_fps':fps,'source_frame_count':decoded,'source_header_frame_count':n,'decoder_header_frame_difference':n-decoded,'duration_s':duration,'audio_start_pts_s':audio_start,'video_start_pts_s':video_start,
          'av_offset_ms':round((audio_start-video_start)*1000,6),'audio_pts_parse_found':match is not None,'valid5fps_frames':len(frames),'frames':metadata,'face_detector':'opencv4.10 haar frontalface_default,320px detection,15% margin','cal_read':False}
    write_json(dest/'metadata.json',meta);return meta

def main():
    torch.set_num_threads(2);cv2.setNumThreads(2);detector_path=Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'
    ascii_copy=Path(tempfile.gettempdir())/('lab9_haar_'+file_hash(detector_path)[:12]+'.xml');ascii_copy.write_bytes(detector_path.read_bytes())
    detector=cv2.CascadeClassifier(str(ascii_copy))
    if detector.empty():raise RuntimeError('Face detector resource could not be loaded; refuse to mark faces missing')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();seen=set();start=time.time()
    write_json(R/'media_preprocessing_protocol.json',{'opencv':cv2.__version__,'face_detector_sha256':file_hash(detector_path),'ffmpeg_executable':ffmpeg,'ffmpeg_sha256':file_hash(ffmpeg),'source':'official full-AV speech only','face_frames':{'size':[224,224],'raw_fps':10,'encoded_fps':5,'face_found_false':'skip encoding, retain metadata'},'cal':'No extraction/decode/features'})
    while True:
        for p in sorted((R/'acquisition').glob('actor_*.json')):
            if p.name in seen:continue
            actor=json.loads(p.read_text());clips=actor['clips']
            if actor['role']!='cal':
                for j,clip in enumerate(clips):
                    process(clip,detector,ffmpeg)
                    if j%20==0:print('PREPROCESS',actor['actor_id'],j,'/60',flush=True)
            seen.add(p.name);write_json(R/'preprocessing_progress.json',{'actor_manifests_seen':sorted(seen),'elapsed_seconds':time.time()-start})
        if len(seen)==24 and (R/'acquisition_complete.json').exists():break
        time.sleep(2)
    write_json(R/'preprocessing_complete.json',{'status':'PASS','actors':24,'decoded_roles':['train','val','test'],'decoded_clips':1320,'cal_decoded':False,'wall_seconds':time.time()-start})

if __name__=='__main__':main()
