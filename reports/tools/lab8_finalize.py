"""Measured counterfactual answers and per-checkpoint reliability profiles."""
from pathlib import Path
import numpy as np
import torch
from reports.tools.lab8_common import *
from src.analysis.reliability_profile import write_profile
from src.reliability.uncertainty import uncertainty

def main():
    e1=read(R/'E1.json');e2=read(R/'E2.json');e3=read(R/'E3.json');ss=read(R/'selective_ood.json');temps=read(R/'temperatures.json');idx=registry();answers=[]
    a=[]
    for arch in ['cnn','vit']:
        rr=[r for r in e1['rows'] if r['model']==arch];a.append({'model':arch,'RUN_ID':rr[0]['RUN_ID'],'fit_RUN_ID':rr[0]['fit_RUN_ID'],
            'ECE_delta':rr[1]['ece']['value']-rr[0]['ece']['value'],'NLL_delta':rr[1]['nll']-rr[0]['nll'],'Brier_delta':rr[1]['brier']-rr[0]['brier'],
            'Tstar':rr[1]['temperature'],'measured_counterexample':rr[1]['ece']['value']<rr[0]['ece']['value'] and (rr[1]['nll']>rr[0]['nll'] or rr[1]['brier']>rr[0]['brier'])})
    answers.append({'question':'Q1','RUN_IDs':[x['RUN_ID'] for x in a]+[x['fit_RUN_ID'] for x in a],'values':a,'answer_ru':'В E1 обе модели улучшили ECE, NLL и Brier; измеренного контрпримера нет. Это не доказывает невозможность: ECE учитывает только максимальную вероятность и усреднение по интервалам, NLL/Brier — весь вектор вероятностей (8.4)-(8.5). На val NLL не растёт, потому что минимизируется именно эта функция, и T=1 явно включена как запасной кандидат. На test такой гарантии нет.'})
    t5=e2['T5'];diffs={}
    for cond in ['clean','shift5']:
        before=t5[cond]['metrics'][1];after=t5[cond]['metrics'][2]
        diffs[cond]={'Tstar':before['temperature'],'T5':after['temperature'],'RUN_ID':t5[cond]['RUN_ID'],'T5_minus_Tstar':{'ECE':after['ece']['value']-before['ece']['value'],**{k:after[k]-before[k] for k in ['accuracy','nll','brier','mean_confidence']}},'all_metrics':t5[cond]['metrics']}
    answers.append({'question':'Q2','RUN_IDs':[t5['fit']['RUN_ID'],t5['clean']['RUN_ID'],t5['shift5']['RUN_ID']],'values':diffs,'answer_ru':'T5 подбирается на occlusion5 val и лучше соответствует известному сдвигу, но перенос на чистые данные не гарантирован. Разности T5-Tstar приведены выше отдельно для ECE/NLL/Brier/accuracy/confidence; отрицательные ECE/NLL/Brier означают улучшение. Классы не меняются. Для заранее неизвестного сдвига такой val нет; результат не является универсальным калибратором.'})
    q3=[];q4=[];q5=[]
    for arch in ['cnn','vit']:
        rr=[r for r in e3['rows'] if r['source_key']==arch+'_1'];eps=config()['profile']['lab8_pgd_eps_255'];fg=next(r for r in rr if r['attack']=='FGSM' and r['eps_255']==eps);pg=next(r for r in rr if r['attack']=='PGD' and r['K']==10);rd=next(r for r in rr if r['attack']=='random')
        q3.append({'model':arch,'random_accuracy':rd['metrics'][0]['accuracy'],'FGSM_accuracy':fg['metrics'][0]['accuracy'],'PGD10_accuracy':pg['metrics'][0]['accuracy'],'prediction_random_gt_FGSM_ge_PGD':rd['metrics'][0]['accuracy']>fg['metrics'][0]['accuracy']>=pg['metrics'][0]['accuracy'],'RUN_IDs':[rd['RUN_ID'],fg['RUN_ID'],pg['RUN_ID']],'epsilon':eps/255})
        q4.append({'model':arch,'RUN_ID':fg['RUN_ID'],'T1':fg['metrics'][0],'Tstar':fg['metrics'][1],'prediction_changes':0,'error_changes':0})
        s=next(r for r in ss if r['source_key']==arch+'_1');before=s['conditions'][0];after=s['conditions'][1];d=outputs(idx['clean_test_'+arch+'_1']);u1=uncertainty(d['logits'],1.)['msp'].numpy();uT=uncertainty(d['logits'],after['temperature'])['msp'].numpy()
        positions_changed=int(np.count_nonzero(np.argsort(u1,kind='stable')!=np.argsort(uT,kind='stable')))
        q5.append({'model':arch,'RUN_ID':s['RUN_ID'],'ID_RUN_ID':s['ID_RUN_ID'],'OOD_RUN_ID':s['OOD_RUN_ID'],'T1':{'AURC':before['measures']['msp']['selective']['AURC'],'AUROC':before['measures']['msp']['ood']['AUROC']},'Tstar':{'temperature':after['temperature'],'AURC':after['measures']['msp']['selective']['AURC'],'AUROC':after['measures']['msp']['ood']['AUROC']},'ID_rank_positions_changed':positions_changed,'class_changes':0})
    answers.append({'question':'Q3','RUN_IDs':[i for r in q3 for i in r['RUN_IDs']],'values':q3,'answer_ru':'Предсказанный заранее порядок проверен на одинаковых test ID. При одинаковом ограничении Linf случайный знак не согласован с градиентом CE; FGSM максимизирует линейное приближение, PGD многократно уточняет направление с проекцией (8.9)-(8.10). Наблюдение относится к этим двум замороженным моделям и epsilon=2/255, а не ко всем возможным шумам.'})
    answers.append({'question':'Q4','RUN_IDs':[r['RUN_ID'] for r in q4],'values':q4,'answer_ru':'Температура изменяет уверенность и ECE, но не исправляет ни одного ошибочного класса: число изменившихся предсказаний и ошибок равно нулю (8.6). Подбор на чистой val не гарантирует калибровку атакованных объектов. Метрики вычислены по одним и тем же сохранённым FGSM logits.'})
    answers.append({'question':'Q5','RUN_IDs':[r['RUN_ID'] for r in q5],'values':q5,'answer_ru':'Температура сохраняет порядок логитов внутри объекта, но не обязана сохранять порядок максимальных вероятностей между объектами в многоклассовой задаче. Поэтому AURC и OOD AUROC могут измениться; измеренные значения и число изменённых позиций указаны выше. Инвариантность AUROC касается строго возрастающего преобразования уже готовой скалярной U (8.13), а softmax(z/T) в общем случае не является одним таким преобразованием MSP для всех векторов z.'})
    write_json(R/'counterfactual_answers.json',answers)
    lines=['# Ответы на пять контрфактических вопросов','']
    for a in answers:
        lines+=['## '+a['question'],'',a['answer_ru'],'','RUN_ID: '+', '.join(a['RUN_IDs']),'','```json',__import__('json').dumps(a['values'],ensure_ascii=False,indent=2),'```','']
    (R/'counterfactual_answers.md').write_text('\n'.join(lines),encoding='utf-8')
    for sk in ['cnn_1','cnn_2','cnn_3','vit_1']:
        s=next(r for r in ss if r['source_key']==sk);seed=int(sk.split('_')[1]);arch=sk.split('_')[0]
        clean=[r for r in e1['rows'] if r['model']==arch] if seed==1 else next(r['clean'] for r in e2['pairs'] if r['seed']==seed)
        if seed>1:
            pair=next(r for r in e2['pairs'] if r['seed']==seed)
            clean=[{**m,'RUN_ID':pair['clean_RUN_ID'],'fit_RUN_ID':temps[sk]['RUN_ID']} for m in clean]
        corr=[r for r in e2['rows'] if r['source_key']==sk];adv=[r for r in e3['rows'] if r['source_key']==sk]
        protocol={'source':read(R/'model_sources.json')[sk],'temperature':temps[sk],'ECE':config()['ece'],'data_protocol':read(R/'data_protocol.json'),'corruptions':config()['corruptions'],'attacks':config()['attacks'],'scope':'CNN seed1 full corruption grid; CNN seed2/3 paired main condition only; attacks seed1 both architectures','all_RUN_IDs':[Path(p).name for k,p in idx.items() if sk in k]}
        result=write_profile(R/'profiles'/sk/'reliability.json',clean=clean,corruption=corr,adversarial=adv,selective={'RUN_ID':s['RUN_ID'],'conditions':s['conditions']},ood={'RUN_ID':s['RUN_ID'],'source_RUN_ID':s['OOD_RUN_ID'],'conditions':s['conditions']},protocol=protocol)
        if not corr:result['undefined'].append({'metric':'corruption grid','data':sk,'reason':'E2 assigned architecture is CNN; ViT corruption grid not required or measured'})
        if not adv:result['undefined'].append({'metric':'adversarial evaluation','data':sk,'reason':'E3 exploratory seed1 only; seed2/3 attack not measured'})
        if seed>1:result['undefined'].append({'metric':'full corruption grid outside assigned level5','data':sk,'reason':'Complete grid exploratory seed1; seed2/3 measured only paired main comparison. Unmeasured, not a missing class definition.'})
        write_json(R/'profiles'/sk/'reliability.json',result)
    print('Five counterfactuals and four reliability profiles generated',flush=True)

if __name__=='__main__':main()
