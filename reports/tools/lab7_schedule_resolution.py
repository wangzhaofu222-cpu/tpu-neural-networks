"""Resolved numerical schedule companions, preserving original checkpoint config hashes."""
from pathlib import Path
import yaml
from src.diffusion.schedules import make_schedule
from src.utils.io import write_json,file_hash


def main():
    count=0
    for p in sorted(Path('runs/lab7').iterdir()):
        path=p/'config.resolved.yaml'
        if not path.exists():continue
        cfg=yaml.safe_load(path.read_text(encoding='utf-8'));s=make_schedule(cfg['diffusion'],diagnostic=True)
        e=cfg.get('evaluation',{})
        result={'RUN_ID':p.name,'original_config_sha256':file_hash(path),'kind':s.kind,'T':s.T,'dtype':'float64',
            'effective_beta_range':[float(s.beta[1:].min()),float(s.beta[1:].max())],
            'linear_beta_arguments':[cfg['diffusion']['beta_min'],cfg['diffusion']['beta_max']],
            'cosine_delta':.008 if s.kind=='cosine' else None,'cosine_beta_cap':.999 if s.kind=='cosine' else None,
            'alpha_bar_0':1.,'alpha_bar_T':float(s.alpha_bar[-1]),'terminal_protocol_valid':s.protocol_valid,
            'original_diffusion_alpha_bar_T_metadata':cfg['diffusion'].get('alpha_bar_T'),
            'original_diffusion_scalar_stale':abs(float(s.alpha_bar[-1])-cfg['diffusion'].get('alpha_bar_T',float(s.alpha_bar[-1])))>1e-15,
            'reverse_variance':cfg['diffusion']['reverse_variance'],'posterior_var_1':float(s.posterior_var[1]),
            'posterior_var_T':float(s.posterior_var[-1]),'sampler':e.get('sampler'),'tau':e.get('tau'),
            'DDIM_eta':e.get('eta'),'clipping':False,
            'note':'beta_min/max in original cosine config are unused linear-schedule arguments; actual cosine range is effective_beta_range. The early shortT diagnostic retained an inherited diffusion.alpha_bar_T scalar, while its evaluation.alpha_bar_T and all actual calculations were correct; original_diffusion_scalar_stale exposes this and this companion resolves it without overwriting provenance. No reverse tau/eta applies to training-only or metric-noise runs.'}
        write_json(p/'schedule.resolved.json',result);count+=1
    print(count,'resolved schedule companions saved',flush=True)


if __name__=='__main__':main()
