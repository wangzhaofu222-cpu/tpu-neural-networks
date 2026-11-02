import numpy as np


def classification(probabilities,labels,classes=8):
    p=np.asarray(probabilities);y=np.asarray(labels);pred=p.argmax(1);cm=np.zeros((classes,classes),dtype=int);np.add.at(cm,(y,pred),1)
    tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*tp,den,out=np.zeros(classes,dtype=float),where=den>0)
    return {'macro_f1':float(f1.mean()),'accuracy':float((y==pred).mean()),'confusion_matrix':cm.tolist(),'per_class_f1':f1.tolist(),'N':len(y),'mean_confidence':float(p.max(1).mean())}


def majority_baseline(train_labels,evaluation_labels,classes=8):
    majority=int(np.bincount(train_labels,minlength=classes).argmax());p=np.eye(classes)[np.full(len(evaluation_labels),majority)]
    return {'majority_class_selected_on_train':majority,**classification(p,evaluation_labels,classes)}


def contribution(full,without_audio,without_video,labels,M0):
    M=classification(full,labels)['macro_f1'];Ra=M-classification(without_audio,labels)['macro_f1'];Rv=M-classification(without_video,labels)['macro_f1'];den=M-M0
    return {'M_AV':M,'M_without_audio':M-Ra,'M_without_video':M-Rv,'Ra':Ra,'Rv':Rv,'Rtilde_a':Ra/den if den>0 else None,'Rtilde_v':Rv/den if den>0 else None,
            'undefined':[] if den>0 else [{'metric':'normalized modality contributions','reason':'M_AV-M0 is nonpositive'}],
            'delta_a':np.abs(np.asarray(full)-np.asarray(without_audio)).sum(1).tolist(),'delta_v':np.abs(np.asarray(full)-np.asarray(without_video)).sum(1).tolist(),'M0':M0}
