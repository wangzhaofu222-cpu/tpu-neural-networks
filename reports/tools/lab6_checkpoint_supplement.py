"""Load every Lab5 checkpoint, including budget files, without modifying any RUN."""
from pathlib import Path
import yaml,torch
from src.models.small_cnn import build_model
from src.utils.io import write_json,digest,file_hash

def main():
    torch.set_num_threads(2);rows=[]
    for f in sorted(Path('runs/lab5').glob('*/checkpoints/*.pt')):
        cfg=yaml.safe_load((f.parent.parent/'config.resolved.yaml').read_text(encoding='utf-8'))
        state=torch.load(f,map_location='cpu',weights_only=False);model=build_model(cfg['model']);model.load_state_dict(state['model'],strict=True)
        assert state['config_hash']==digest(cfg)
        rows.append({'path':str(f),'sha256':file_hash(f),'RUN_ID':f.parent.parent.name,'purpose':cfg['experiment']['purpose'],'load_status':'PASS','config_hash_matches':True})
    assert len(rows)==286
    write_json('experiments/lab5_verification/all_checkpoints_loadability.json',{'status':'PASS','count':len(rows),'includes_budget_checkpoints':True,'optimizer_updates_executed':0,'checkpoints':rows})
    print('All286Lab5checkpoints loaded with strict model/config matching; no mutation')
if __name__=='__main__':main()
