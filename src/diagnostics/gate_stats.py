import numpy as np


def gate_statistics(gates,labels,correct):
    if gates is None:return {'status':'NOT_APPLICABLE','reason':'model has no gate'}
    g=np.asarray(gates);g=g.reshape(len(g),-1).mean(1);y=np.asarray(labels);c=np.asarray(correct,dtype=bool)
    return {'status':'DEFINED','mean':float(g.mean()),'std':float(g.std(ddof=1)) if len(g)>1 else 0.,'by_emotion':{str(k):float(g[y==k].mean()) if (y==k).any() else None for k in range(8)},'correct_mean':float(g[c].mean()) if c.any() else None,'incorrect_mean':float(g[~c].mean()) if (~c).any() else None,'convention':'1 video, 0 audio'}
