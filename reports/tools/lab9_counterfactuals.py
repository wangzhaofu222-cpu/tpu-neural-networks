import numpy as np
from reports.tools.lab9_common import *

def main(reuse_double=False,destination=None):
    e1=read(R/'E1_exploration.json');e2=read(R/'E2.json');main=read(R/'E1_main.json');selection=e1['selection'];U=selection['U'];F=selection['F'];strategy=selection['specs'][U]['strategy'];width=selection['controls'][F][strategy]['match']['width']
    if reuse_double:double=read(R/'counterfactuals.json')['double_control']
    else:
        if (R/'counterfactuals.json').exists():raise RuntimeError('Archived counterfactual training exists; replay answers without retraining instead')
        spec={'path':train('Q2_double_unimodal_control',strategy,width=width*2),'strategy':strategy};double=evaluate('Q2_double_control_test',spec)
    old=e1['test'][U];f=e1['test'][F];gated=e2['models']['gated'];c=gated['contribution'];blur=e2['supplementary']['blur'];s=e2['shuffle'];noise=e2['noise'];rows=[]
    rows.append({'question':1,'RUN_ID':[gated['RUN_ID'],e1['test']['video']['RUN_ID'],e1['test']['audio']['RUN_ID']],'values':{'M_video':e1['test']['video']['macro_f1'],'M_audio':e1['test']['audio']['macro_f1'],'Rtilde_a':c['Rtilde_a'],'Rtilde_v':c['Rtilde_v'],'mean_delta_a':float(np.mean(c['delta_a'])),'mean_delta_v':float(np.mean(c['delta_v']))},'answer':'Слабая модальность не обязана использоваться. Ее вклад определяется вмешательством в ту же модель; качество отдельного классификатора этого не доказывает. Ненулевые L1-изменения показывают зависимость вероятностей, но не гарантируют полезный прирост F1. Вывод ограничен заменителем zero и данным разбиением.'})
    # Main control is validation-selected U; max over both exploratory matched controls is kept separately.
    original_gctrl=e1['gains'][F]['Gctrl'];other=[e1['test'][selection['controls'][F][st]['name']]['macro_f1'] for st in ['video','audio'] if st!=strategy];new_gctrl=f['macro_f1']-max([double['macro_f1']]+other)
    rows.append({'question':2,'RUN_ID':[double['RUN_ID'],old['RUN_ID'],f['RUN_ID']],'values':{'control_strategy':strategy,'old_width':width,'new_width':2*width,'old_control_F1':old['macro_f1'],'double_control_F1':double['macro_f1'],'F_F1':f['macro_f1'],'old_Gctrl':original_gctrl,'new_Gctrl':new_gctrl,'new_F_minus_doubled_U':f['macro_f1']-double['macro_f1']},'answer':'Контроль действительно переобучен с удвоенной скрытой шириной и теми же200эпохами. Новый Gctrl использует максимум обновленного контроля и прежнего контроля другой модальности. Сам по себе выигрыш G не отделяет комплементарность от емкости; основной парный вывод приведен отдельно. Это единичный дополнительный seed, не трехseedовая статистика.'})
    rows.append({'question':3,'RUN_ID':[s['RUN_ID'],gated['RUN_ID']],'values':{k:s[k] for k in ['mean_delta_shuffle','mean_delta_mask','changed_class_shuffle','changed_class_mask','fixed_points']}|{'Ra':c['Ra']},'answer':'Почти нулевое изменение после перемешивания указывало бы на слабую зависимость от аудио; этот условный тезис проверяется фактическими величинами выше, а не принимается заранее. Zero и чужое реальное аудио задают разные вмешательства, поэтому Ra и L1 могут расходиться.'})
    bc=blur['contribution'];rows.append({'question':4,'RUN_ID':[gated['RUN_ID'],blur['RUN_ID']],'values':{'clean_F1':gated['macro_f1'],'blur_F1':blur['macro_f1'],'clean_gate':gated['gate']['mean'],'blur_gate':blur['gate']['mean'],'blur_Rv':bc['Rv'],'blur_mean_delta_v':float(np.mean(bc['delta_v']))},'answer':'Размытие уровня1 ЛР8 (sigma=.5,kernel5) применено к кадрам с повторным кодированием. Стабильный F1 сам по себе не доказывает устойчивость визуального пути: проверяются Rv и delta_v той же модели. Нулевой вклад означал бы возможное игнорирование; вывод не переносится на более сильное размытие.'})
    rows.append({'question':5,'RUN_ID':[r['RUN_ID'] for k in noise for r in noise[k]],'values':{k:[{'snr':r['snr_db'],'F1':r['macro_f1'],'gate_video_weight':r['gate']['mean']} for r in v] for k,v in noise.items()},'answer':'Шлюз не обязан монотонно увеличивать вес видео при снижении SNR: обучение на чистом аудио не задает такой целевой функции. Modality dropout учит отсутствию, а не всему семейству шумов. Кривые получены из одного seed и фиксированных реализаций шума; направление описывается измерениями без принудительного тренда.'})
    weak='video' if e1['test']['video']['macro_f1']<e1['test']['audio']['macro_f1'] else 'audio'
    relevant=c['Rv'] if weak=='video' else c['Ra'];delta=np.mean(c['delta_v'] if weak=='video' else c['delta_a'])
    rows[0]['answer']+=f' В этих данных слабее {weak}: R={relevant:.6f}, среднее delta={delta:.6f}. Удаление '+('ухудшает F1' if relevant>0 else ('улучшает F1' if relevant<0 else 'не меняет F1'))+'; зависимость вероятностей '+('наблюдается' if delta>1e-7 else 'численно не обнаружена')+'.'
    rows[1]['answer']+=f' Измеренное изменение Gctrl: {new_gctrl-original_gctrl:+.6f}; '+('преимущество F остается положительным.' if new_gctrl>0 else 'после этого контроля положительный выигрыш F не показан.')
    rows[2]['answer']+=f" Средний L1 shuffle={s['mean_delta_shuffle']:.6f}, mask={s['mean_delta_mask']:.6f}; доля смены класса {s['changed_class_shuffle']:.6f} и {s['changed_class_mask']:.6f}. При Ra={c['Ra']:.6f} удаление "+('вредит качеству.' if c['Ra']>0 else 'не демонстрирует полезный вклад аудио по F1.')
    rows[3]['answer']+=f" Здесь F1 изменился на {blur['macro_f1']-gated['macro_f1']:+.6f}; Rv после blur={bc['Rv']:.6f}, mean delta_v={np.mean(bc['delta_v']):.6f}. Это непосредственно отделяет зависимость выхода от стабильности итоговой F1."
    for key,rs in noise.items():
        gates=np.array([r['gate']['mean'] for r in rs]);rows[4]['answer']+=f' {key}: g(-5)−g(30)={gates[-1]-gates[0]:+.6f}; рост веса видео на всех соседних шагах сетки '+('наблюдается' if np.all(np.diff(gates)>=0) else 'не наблюдается')+'.'
    out=Path(destination) if destination else R;out.mkdir(parents=True,exist_ok=True)
    write_json(out/'counterfactuals.json',{'answers':rows,'double_control':double,'main_difference':main['decision']})
    text='# Ответы на пять контрфактических вопросов\n\n'
    for row in rows:text+=f"## Q{row['question']}\n\nRUN_ID: "+', '.join(row['RUN_ID'])+'\n\n```json\n'+json.dumps(row['values'],ensure_ascii=False,indent=2)+'\n```\n\n'+row['answer']+'\n\n'
    (out/'counterfactuals.md').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
