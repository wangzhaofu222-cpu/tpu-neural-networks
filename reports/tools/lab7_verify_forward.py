"""Reconstruct registered forward draws once; retain raw draws without rewriting results."""
import torch
from src.diffusion.schedules import make_schedule
from src.diffusion.forward import q_sample
from src.data.gmm2d import roles_and_stats,standardize
from src.utils.io import write_json,file_hash
from reports.tools.lab7_common import R,base,read


def main():
    torch.set_num_threads(2);e1=read(R/'E1.json');roles,stats=roles_and_stats(base());train=standardize(roles['train'],stats)
    samples={};checks=[]
    for kind in ['linear','cosine']:
        s=make_schedule({**base()['diffusion'],'kind':kind});g=torch.Generator().manual_seed(7301)
        for row in e1['theory'][kind]['forward_moments']:
            noise=torch.randn(10000,2,generator=g,dtype=torch.float64)
            xt,_=q_sample(train[:1].expand(10000,2),row['t'],s,noise)
            checks.append({'schedule':kind,'t':row['t'],'PASS':xt.mean(0).tolist()==row['mean_measured'] and xt.var(0).tolist()==row['variance_measured']})
            samples[kind+'_'+str(row['t'])]=xt
            torch.randn(train.shape,generator=g,dtype=torch.float64) # consume registered full-train cloud innovations
    out=R/'verification';out.mkdir(exist_ok=True)
    torch.save(samples,out/'fixed_x0_samples_reconstructed.pt')
    write_json(R/'forward_regeneration_audit.json',{'status':'PASS' if all(c['PASS'] for c in checks) else 'FAIL','checks':checks,
        'raw_reconstructed_path':str(out/'fixed_x0_samples_reconstructed.pt'),'sha256':file_hash(out/'fixed_x0_samples_reconstructed.pt'),
        'note':'Registered seed and exact stream usage reconstruct all first-draw aggregate measurements; original E1.json is unchanged.'})
    print('Forward reconstruction',all(c['PASS'] for c in checks),flush=True)


if __name__=='__main__':main()
