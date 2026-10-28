"""Save eight actual train images and independent configured SimCLR views."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import load_config
from src.data.ssl_data import bundle
from src.ssl.views import Views
from src.ssl.augmentations import augmentation_hash
from src.utils.io import write_json

def main():
    c=load_config('configs/lab4.yaml');images,split,mean,std,_=bundle(c['data']['root'],c['data']['split_path'])
    ids=split['roles']['train'][:8];seed=c['profile']['base_seed']+9010
    ds=Views(images,ids,mean,std,c['augmentations'],seed)
    output=Path('experiments/lab4/figures');output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(8,3,figsize=(6.8,15))
    for i,index in enumerate(ids):
        a,b,_=ds.two_views(i);views=[images[index]]
        for x in [a,b]:views.append((x.permute(1,2,0).numpy()*np.array(std)+np.array(mean)).clip(0,1))
        for j,x in enumerate(views):
            axes[i,j].imshow(x);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            if i==0:axes[i,j].set_title(['Original train image','Independent view1','Independent view2'][j])
        axes[i,0].set_ylabel(f'Index{index}',fontsize=9)
    fig.suptitle('Configured T: random resized crop +colorjitter +horizontalflip',fontsize=11)
    fig.tight_layout();fig.savefig(output/'augmentation_views.png',dpi=140);plt.close(fig)
    write_json('experiments/lab4/augmentation_gallery.json',{'role':'train','image_indices':ids,'view_seed':seed,
        'augmentation_hash':augmentation_hash(c['augmentations']),'labels_read':False,'output':(output/'augmentation_views.png').as_posix()})
if __name__=='__main__':main()
