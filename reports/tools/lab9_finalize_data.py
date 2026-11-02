"""Audit genuine archive identities and compute train-only mel statistics."""
import json
from pathlib import Path
import torch
from src.utils.io import write_json,digest,file_hash
from src.multimodal.identity import validate_clip_identity

R=Path('experiments/lab9')
def main():
    assert (R/'acquisition_complete.json').is_file()
    assert (R/'preprocessing_complete.json').is_file()
    archives=[json.loads(p.read_text()) for p in sorted((R/'acquisition').glob('actor_*.json'))]
    assert len(archives)==24
    clips=[]
    for a in archives:
        assert a['official_md5']==a['verified_md5'] and len(a['clips'])==60
        for c in a['clips']:
            if c['role']=='cal':
                assert c['media_path'] is None
                assert not (Path('data/lab9/preprocessed')/c['clip_id']).exists()
                clips.append(c)
            else:
                m=json.loads((Path('data/lab9/preprocessed')/c['clip_id']/'metadata.json').read_text())
                assert m['clip_id']==c['clip_id'] and m['actor_id']==c['actor_id'] and m['role']==c['role']
                assert m['audio_pts_parse_found'] and m['audio_samples']>0
                assert file_hash(m['audio_path'])==m['audio_sha256']
                assert file_hash(m['video_frames_path'])==m['video_frames_sha256']
                clips.append(m)
    assert len({c['clip_id'] for c in clips})==1440
    validate_clip_identity(clips,json.loads((R/'planned_actor_split.json').read_text())['roles'])
    sizes={r:sum(c['role']==r for c in clips) for r in ['train','val','cal','test']}
    assert sizes==dict(train=840,val=180,cal=120,test=300)
    sums=torch.zeros(64,dtype=torch.float64);squares=sums.clone();n=0
    for c in clips:
        if c['role']!='train':continue
        x=torch.load(c['log_mel_path'],weights_only=True).double()
        assert x.shape[0]==64 and torch.isfinite(x).all()
        sums+=x.sum(1);squares+=x.square().sum(1);n+=x.shape[1]
    mean=sums/n;std=(squares/n-mean.square()).clamp_min(1e-12).sqrt()
    write_json(R/'mel_normalization.json',{'mean':mean.tolist(),'std':std.tolist(),'fit_role':'train','frames':n,'clip_ids':[c['clip_id'] for c in clips if c['role']=='train'],'ddof':0})
    manifest={'status':'PASS','clips':clips,'split_sizes':sizes,'cal_read':False,'teacher_split_status':'TEACHER_SPLIT_UNVERIFIED','raw_identity_hash':digest({a['actor_id']:a['sha256'] for a in archives}),'split_fingerprint':file_hash(R/'planned_actor_split.json'),'class_counts':{str(i):sum(c['emotion']==i for c in clips) for i in range(8)},'noncal_face_missing_clips':[c['clip_id'] for c in clips if c['role']!='cal' and c['valid5fps_frames']==0]}
    write_json(R/'data_manifest.json',manifest)
    (R/'data_audit.md').write_text('# Lab9 数据审计\n\n24个官方 speech归档全部通过官方MD5校验，并保存SHA256和ZIP身份清单；1440个唯一AV片段/24演员/8类别。train840,val180,cal120,test300；演员完全隔离。用户授权1729确定性性别分层后备split，保留TEACHER_SPLIT_UNVERIFIED。cal只枚举ZIP身份元数据，不提取/解码/编码媒体。\n\n非cal共1320个片段，16kHz单声道和64频带logmel；真实720p/29.97FPS视频生成224脸帧10FPS并按原索引5FPS编码。face_found=false跳过并保留身份/时间/有效掩码。每片段记录实际音视频PTS偏移、解码帧数和容器头帧数。一帧差异被完整保留，更大差异拒绝。mel标准化只用train有效时间帧，后续clip特征标准化只用train均值。\n\n教师Drive完整14页文本已通过连接器读取，俄语正文字符顺序完全匹配本地讲义；数学重音/上下标抽取顺序有差异，关键公式已视觉核对。远程原始字节下载403，因此不声称PDF逐字节身份已核对。\n\n检测器、FFmpeg、媒体与缓存哈希和全部计数见相邻JSON。原始下载中断及预处理开发失败记录保留，未将不完整归档误标为哈希通过。\n',encoding='utf-8')
    print('GENUINE DATA AUDIT PASS',sizes,'missing faces',len(manifest['noncal_face_missing_clips']),flush=True)

if __name__=='__main__':main()
