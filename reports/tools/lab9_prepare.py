import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import yaml
from src.profile_gen import generate
from src.utils.io import write_json,file_hash

R=Path('experiments/lab9')
def main():
    R.mkdir(exist_ok=True);sid=json.loads(Path('.private/lab4_student_id.json').read_text())['student_id'];profile=generate(sid)
    p={k:v for k,v in profile.items() if k=='profile' or k.startswith('lab9_')};p.update(calculation_input='private existing student_id; public SHA256 only',lab9_hash=hashlib.sha256((sid+':lab9').encode()).hexdigest(),lab_pick_source_sha256=file_hash('src/profile_gen.py'),profile_conflict_status='UNVERIFIED: teacher fields conflict; user previously authorized visual-source precedence')
    write_json(R/'profile.json',p)
    selected=json.loads(Path('experiments/lab2/selected_lr.json').read_text())['cnn'];root=Path(selected['path']);cfg=yaml.safe_load((root/'config.resolved.yaml').read_text(encoding='utf-8'));weight=root/'checkpoints/last.pt';manifest=json.loads((root/'manifest.json').read_text())
    write_json(R/'encoder_sources.json',{'visual':{'RUN_ID':root.name,'path':str(weight),'sha256':file_hash(weight),'config':cfg,'normalization':manifest['normalization'],'effective_architecture':'cifar_resnet18','feature_dim':512,'input_size':224,'checkpoint_original_input_size':32,'domain_limitation':'CIFAR10 -> RAVDESS faces; adaptive spatial pooling permits224 but distribution/scale mismatch retained','selection':'Existing Lab2 validation-selected full-fraction CNN; no RAVDESS test used'},'audio':{'status':'BLOCKED','reason':'Requires genuine train data and val-selected AudioCNN training','feature_dim':128,'time_downsample_r':1}})
    hist={'Lab1_E2_growth_bound':'FAIL_RETAINED','Lab3_E2_raw_memory_intervals':'FAIL_RETAINED','Lab4_Q2':'BLOCKED','Lab5_course_split':'UNVERIFIED','Lab5_E2_magnitude':'FAIL','Lab6_E1_mixing':'FAIL','Lab8_E2_positive_prediction':'FAIL; reverse difference established','Lab8_initial_CPU_GPU_audit':'FAIL_RETAINED; deterministic audit settings corrected, final827PASS'}
    write_json(R/'historical_states.json',hist)
    config={'experiment':{'name':'lab9_multimodal','lab':9,'seed':1,'deterministic':True,'purpose':'development','role':'primary'},'profile':{**p['profile'],**{k:v for k,v in p.items() if k.startswith('lab9_')}},
            'data':{'name':'multimodal_features','root':'data/lab9','manifest_path':'experiments/lab9/data_manifest.json','split_path':'experiments/lab9/planned_actor_split.json','train_fraction':1.,'num_workers':0,'cal_read':False},
            'model':{'name':'multimodal_fusion','strategy':'gated','classes':8,'visual_dim':512,'audio_dim':128,'hidden_width':128,'fusion_dim':128,'gate':p['lab9_gate'],'p_drop':0.},
            'optimizer':{'name':'adamw','lr':.001,'weight_decay':.0001,'momentum':.9},'scheduler':{'name':'constant','unit':'epoch','warmup_epochs':0,'step_size':100,'gamma':1.},
            'train':{'epochs':200,'batch_size':64,'amp':False,'accumulation_steps':1,'early_stopping_patience':30,'selection_metric':'macro_f1'},
            'logging':{'backend':'local','run_root':'runs/lab9','save_every':1,'console_every_steps':1000},'runtime':{'device':'cpu','threads':2,'max_seconds':1800,'dependency_lock':'requirements-lab9.lock.txt'},
            'audio':{'sample_rate':16000,'n_mel':64,'n_fft':400,'hop_length':160,'center':True,'epsilon':1e-10,'mel_scale':'HTK','r':1,'max_seconds':8.,'channels':[16,32,64,128],'train_epochs':100,'patience':15,'batch_size':16},
            'alignment':{'video_fps_raw':10,'video_fps':5,'window_samples':3200,'integer_samples':True,'av_offset_ms':'measured per clip','audio_shift_ms':0},
            'null_token':'zero','snr_grid':[30,20,10,0,-5],'pair_count_per_protocol':300,'data_split_status':'TEACHER_SPLIT_UNVERIFIED'}
    Path('configs/lab9.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    write_json(R/'data_manifest.json',{'status':'BLOCKED','reason':'Official authorized acquisition/preprocessing not yet complete','expected_clips':1440,'expected_actors':24,'expected_classes':8,'source':'https://zenodo.org/records/1188976','teacher_split_status':'TEACHER_SPLIT_UNVERIFIED','cal_read':False,'clips':[]})
    (R/'data_audit.md').write_text('# Lab9 数据来源审计\n\n本地起始未发现 RAVDESS 或教师专用演员 split。用户已授权下载官方24个 speech归档，总13,357,344,723字节；逐包校验MD5，保留原ZIP。当前数据取得尚未完成，科学实验 BLOCKED。planned_actor_split.json为确定性、性别分层14/3/2/5备选划分，明确 TEACHER_SPLIT_UNVERIFIED，不冒充教师划分。cal只做身份元数据清单，不提取媒体、解码或特征。\n\n教师 Google Drive 页面暂无法经浏览工具访问；项目已有14页原PDF完整重读，版本2026-10-08 редакция4，关键公式已视觉核对。官方原始数据为720p/48kHz，需要自行生成224脸帧10FPS和16kHz单声道，不能冒称已有教师预处理包。\n',encoding='utf-8')
    print({k:p[k] for k in ['lab9_visual_encoder','lab9_encoder','lab9_p_drop','lab9_gate']})

if __name__=='__main__':main()
