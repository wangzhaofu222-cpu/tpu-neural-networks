"""Verify the registered public late class against actual frozen functional outputs."""
from pathlib import Path
import torch,yaml
from reports.tools.lab9_common import R,read,features
from src.models.small_cnn import build_model
from src.utils.io import write_json

@torch.no_grad()
def main():
    torch.set_num_threads(2);selection=read(R/'E1_validation_selection.json');x=features()['test']['x'];configs={};checks=[]
    for key in ['late05','late_val']:
        spec=selection['specs'][key];v=Path(spec['video_path']);a=Path(spec['audio_path']);vc=yaml.safe_load((v/'config.resolved.yaml').read_text(encoding='utf-8'))['model'];ac=yaml.safe_load((a/'config.resolved.yaml').read_text(encoding='utf-8'))['model']
        cfg={'name':'multimodal_late','video_model':vc,'audio_model':ac,'video_checkpoint':str(v/'checkpoints/best.pt'),'audio_checkpoint':str(a/'checkpoints/best.pt'),'weight':spec['weight']};configs[key]=cfg;m=build_model(cfg).eval()
        r=read(R/'E1_exploration.json')['test'][key];raw=torch.load(Path('runs/lab9')/r['RUN_ID']/'predictions.pt',weights_only=False)
        for mask,field in [((1,1),'probabilities'),((1,0),'no_audio'),((0,1),'no_video')]:
            p=m(visual=x[:,:512],audio=x[:,512:640],mask=mask);error=float((p-raw[field]).abs().max());checks.append({'strategy':key,'mask':list(mask),'maximum_error':error,'status':'PASS' if torch.equal(p,raw[field]) else 'FAIL'})
    write_json(R/'late_model_configs.json',configs);write_json(R/'late_registry_equivalence.json',{'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL','checks':checks,'scope':'Public class/factory adapter verified after experiments against original saved functional late predictions; bitwise equality for both lambda settings and all three masks. No weights/outputs retrained or changed.'});print(checks)

if __name__=='__main__':main()
