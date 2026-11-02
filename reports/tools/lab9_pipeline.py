"""Run bounded stages; deliberately stop at the real preregistration boundary."""
import argparse,time
from pathlib import Path
from reports.tools.lab9_common import R,read,train
from src.utils.io import file_hash,write_json

def prepare():
    from reports.tools.lab9_finalize_data import main as finalize
    from reports.tools.lab9_features import finish
    while not (R/'preprocessing_complete.json').exists() or not (R/'visual_features_complete.json').exists():time.sleep(10)
    finalize()
    sources=read(R/'encoder_sources.json')
    if sources['audio']['status']!='PASS':
        path=Path(train('AudioCNN_train_val_selected','audio',audio=True));checkpoint=path/'checkpoints/best.pt'
        sources['audio']={'status':'PASS','run_path':str(path),'RUN_ID':path.name,'path':str(checkpoint),'sha256':file_hash(checkpoint),'feature_dim':128,'time_downsample_r':1,'selection':'val macro-F1; max100epochs/patience15; no test/cal access','normalization_sha256':file_hash(R/'mel_normalization.json')};write_json(R/'encoder_sources.json',sources)
    finish()
    from reports.tools.lab9_e1 import explore
    explore()
    print('STOPPED AS REQUIRED: REAL hypothesis Git commit must precede main stage',flush=True)

def after_prereg():
    from reports.tools.lab9_e1 import main_experiment
    from reports.tools.lab9_e2 import main as e2
    from reports.tools.lab9_e3 import main as e3
    from reports.tools.lab9_counterfactuals import main as cf
    from reports.tools.lab9_plots import main as plots
    main_experiment();e2();e3();cf();plots()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','after_prereg']);a=parser.parse_args();prepare() if a.stage=='prepare' else after_prereg()
