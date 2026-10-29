"""NumPy mode/count oracle independent of the PyTorch Lab7 evaluator."""
from pathlib import Path
import numpy as np
import torch,yaml
from src.utils.io import write_json


def main():
    root=Path('experiments/lab7');idx=__import__('json').loads((root/'run_index.json').read_text());checks=[]
    for key,row in idx['evaluations'].items():
        p=Path('runs/lab7')/row['RUN_ID'];cfg=yaml.safe_load((p/'config.resolved.yaml').read_text());d=cfg['data']
        centers=np.array([[np.cos(2*np.pi*m/d['M']),np.sin(2*np.pi*m/d['M'])] for m in range(d['M'])])
        def freq(points):
            distances=np.sqrt(((points[:,None,:]-centers[None,:,:])**2).sum(2));labels=distances.argmin(1)
            labels[distances.min(1)>=3*d['rho']]=d['M']
            return np.bincount(labels,minlength=d['M']+1)/len(points)
        reference=torch.load(root/'metrics'/str(d['pi0'])/'reference100000.pt',weights_only=False).numpy()
        generated=torch.load(p/'generated_samples.pt',weights_only=False).numpy()
        rf=freq(reference);gf=freq(generated);Dmode=np.abs(rf-gf).sum()/2;cov=np.mean(gf[:-1]>=rf[:-1]/2)
        checks.append({'key':key,'RUN_ID':row['RUN_ID'],'PASS':abs(Dmode-row['Dmode'])<1e-12 and cov==row['Cov'] and np.allclose(gf,row['mode_frequencies'],atol=1e-15,rtol=0)})
    write_json(root/'independent_mode_audit.json',{'status':'PASS' if all(r['PASS'] for r in checks) else 'FAIL','checks':checks,'implementation':'independent NumPy distances,argmin,bincount;100kMCreference;outlier included'})
    print(len(checks),'independent mode checks;',sum(not r['PASS'] for r in checks),'failures',flush=True)


if __name__=='__main__':main()
