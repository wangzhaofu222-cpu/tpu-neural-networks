import time,json
from pathlib import Path
import numpy as np
import torch
from reports.tools.lab8_common import R,source_model,role_data,evaluate,read
from src.attacks import fgsm,pgd
from src.utils.io import write_json
from src.utils.reproducibility import seed_all

def main():
    seed_all(1,True,2);measurements=[];x,y,_=role_data('test');x=x[:64].cuda().float()/255;y=y[:64].cuda()
    for arch in ['cnn','vit']:
        m,source=source_model(arch+'_1');torch.cuda.reset_peak_memory_stats();times={}
        for mode in ['forward','fgsm','pgd10']:
            vals=[]
            for repeat in range(4):
                torch.cuda.synchronize();start=time.perf_counter()
                if mode=='forward':
                    with torch.no_grad():m(x)
                elif mode=='fgsm':fgsm(m,x,y,2/255)
                else:
                    from torch.nn.attention import sdpa_kernel,SDPBackend
                    with sdpa_kernel(SDPBackend.MATH):pgd(m,x,y,2/255,K=10,R=1)
                torch.cuda.synchronize()
                if repeat:vals.append(time.perf_counter()-start)
            times[mode]={'median_seconds_per64':float(np.median(vals)),'seconds_per10000':float(np.median(vals)*10000/64),'repeats':vals}
        measurements.append({'model':arch,'times':times,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'device':torch.cuda.get_device_name(),'batch_size':64})
        print(measurements[-1],flush=True)
    # Actual interrupted/resumed shard path is tested by replaying a completed smoke run.
    p=evaluate('smoke_cnn','cnn_1',purpose='smoke',limit=128);before=(p/'outputs.pt').read_bytes()
    evaluate('smoke_cnn','cnn_1',purpose='smoke',limit=128)
    assert before==(p/'outputs.pt').read_bytes()
    try:evaluate('smoke_resume_cnn','cnn_1',purpose='smoke',limit=192,interrupt_after_batches=2)
    except InterruptedError:pass
    from reports.tools.lab8_common import registry,outputs
    q=Path(registry()['smoke_resume_cnn']);shards={str(f):f.read_bytes() for f in (q/'batches').glob('*.pt')}
    evaluate('smoke_resume_cnn','cnn_1',purpose='smoke',limit=192)
    assert all(Path(f).read_bytes()==v for f,v in shards.items())
    assert torch.equal(outputs(p)['logits'],outputs(q)['logits'][:128])
    write_json(R/'budget.json',{'measurements':measurements,'smoke_RUN_ID':p.name,'completed_run_reuse_verified':True,'interrupted_shard_resume_verified':True,'resume_RUN_ID':q.name,'new_classifier_training':False,
        'estimated_inference_attack_seconds':sum(r['times']['forward']['seconds_per10000']*35+r['times']['fgsm']['seconds_per10000']*5+r['times']['pgd10']['seconds_per10000']*5 for r in measurements),
        'protocol_decision':'Keep full test 10000 for both architectures, batch64, all mandatory attack strengths; no new model training.'})

if __name__=='__main__':main()
