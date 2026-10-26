import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    records=[json.loads(line) for line in (a.run/'metrics.jsonl').read_text().splitlines()]
    manifest=json.loads((a.run/'manifest.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    epochs=[r['epoch'] for r in records]
    for ax,columns in zip(axes,[['train_loss','validation_loss'],['train_metric','validation_metric']]):
        for name in columns:
            ax.plot(epochs,[r[name] for r in records],marker='o',label=name)
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(f"{manifest['purpose']} | {a.run.name}",fontsize=9)
    out=a.run/'plots'
    out.mkdir(exist_ok=True)
    fig.savefig(out/'learning_curves.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
