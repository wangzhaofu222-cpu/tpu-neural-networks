from pathlib import Path
import json,re,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'答辩资料_Lab0-9_20261111';KIT=OUT/'05_实验过程'
FLOW=[
['固定数据划分和seed → 训练SmallCNN → 保存完整状态 → 独立重复/中断恢复 → 比较全部loss轨迹与参数','Фиксировать split и seed → обучить SmallCNN → сохранить полное состояние → повторить/resume → сравнить траектории и параметры.'],
['搭建不同深度与初始化 → 逐层读取激活 → 反向传播并监测梯度 → 固定预算训练 → 比较残差与普通网络','Построить сети разной глубины и инициализации → измерить активации и градиенты → обучить при фиксированном бюджете → сравнить residual и plain.'],
['固定数据比例和更新预算 → 训练CNN/ViT → val选学习率 → test评价干净与扰动图像 → 三seed配对','Зафиксировать долю данных и обновления → обучить CNN/ViT → выбрать LR по val → оценить clean и искажения → сравнить три seed.'],
['生成Copy序列 → 加入不同位置编码 → 训练2000更新 → 测试更长序列 → 检查注意力与显存','Сгенерировать copy → добавить позиционный код → обучить 2000 обновлений → проверить новые длины → исследовать attention и память.'],
['同图生成两个增强视图 → 对比预训练 → 冻结编码器 → 训练线性probe → 消融增强与标签比例','Два вида изображения → контрастное pretraining → заморозить encoder → обучить linear probe → исключать аугментации и менять долю меток.'],
['加载Cora特征/图/masks → 训练MLP/GCN/GAT → 保度重连对照 → 深度与残差 → 区分过平滑和优化困难','Cora: признаки, граф и masks → MLP/GCN/GAT → перестановка рёбер с сохранением степеней → глубина/residual → диагностика.'],
['二维目标分布 → 检查Langevin步长和混合 → 训练可逆Flow并选深度 → 在困难目标上比较生成分布','Двумерная цель → проверить Langevin и смешивание → обучить Flow и выбрать глубину → сравнить генерацию на трудной цели.'],
['不等权GMM与train标准化 → 正向加噪 → 训练噪声预测器 → DDPM/DDIM采样 → 精确score对照及稀有模式频率','Неравновесная GMM и train-стандартизация → зашумление → обучение predictor → DDPM/DDIM → точный score и редкая мода.'],
['冻结Lab2分类器 → 仅用val拟合温度 → test测ECE/NLL → 输入扰动与攻击 → risk–coverage及OOD','Заморозить классификатор ЛР2 → подобрать T на val → ECE/NLL на test → искажения/атаки → risk–coverage/OOD.'],
['按演员划分 → 人脸帧与log-mel → 冻结编码器特征 → 训练融合头 → 参数匹配控制 → 缺失/噪声/冲突','Разделить актёров → лица и log-mel → замороженные признаки → fusion → параметрический контроль → пропуск/шум/конфликт.']]
def main():
    labs=json.loads((KIT/'展示索引.json').read_text('utf-8'));items=[]
    demo=json.loads((KIT/'现场演示/演示核查.json').read_text('utf-8'))
    for l in labs:
        n=l['n'];m=l['model'];step=max(1,len(l['loss'])//600);loss=l['loss'][::step]
        if loss and loss[-1]!=l['loss'][-1]:loss.append(l['loss'][-1])
        training_folder=(KIT/'训练来源'/f'lab{n}'/m['training_run']).relative_to(OUT).as_posix()
        items.append({'n':n,'title':l['title'],'flow':FLOW[n],'runs':l['runs'],'loss':loss,'raw_loss_count':len(l['loss']),'plots':l['plots'],'model':m,'training_folder':training_folder,'demo':demo['runs'][n]})
    html=(Path(__file__).parent/'demo_template.html').read_text('utf-8')
    html=html.replace('/*DATA*/',json.dumps(items,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    (OUT/'实验过程展示.html').write_text(html,'utf-8')
    print('Offline page written: 10 labs,',sum(len(x['runs']) for x in labs),'registered records')
if __name__=='__main__':main()
