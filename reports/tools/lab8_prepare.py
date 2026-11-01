"""Select existing, genuinely matched checkpoints without observing Lab8 test metrics."""
import json
from pathlib import Path
import yaml
from src.profile_gen import generate
from src.utils.io import write_json,file_hash
from src.corruptions.image import PARAMETERS

def main():
    R=Path('experiments/lab8');R.mkdir(exist_ok=True)
    profile=generate(json.loads(Path('.private/lab4_student_id.json').read_text())['student_id'])
    write_json(R/'profile.json',{k:v for k,v in profile.items() if k=='profile' or k.startswith('lab8_')})
    index=json.loads(Path('experiments/lab2/checkpoint_index.json').read_text());sources={}
    for arch in ['cnn','vit']:
        for seed in [1,2,3]:
            label=f'E1_{arch}_r0.05_j1' if seed==1 else f'E3_{arch}_j{seed}'
            row=next(x for x in index if x['label']==label and x['path'].endswith('last.pt'))
            cfg=yaml.safe_load((Path(row['path']).parent.parent/'config.resolved.yaml').read_text(encoding='utf-8'))
            assert row['fraction']==.05 and row['seed']==seed and row['steps']==10000 and file_hash(row['path'])==row['sha256']
            sources[f'{arch}_{seed}']={**row,'cfg':cfg,'source_summary':json.loads((Path(row['path']).parent.parent/'summary.json').read_text())}
    write_json(R/'model_sources.json',sources)
    cfg={'experiment':{'lab':8,'seed':1,'deterministic':True,'purpose':'formal'},'profile':{**profile['profile'],**{k:v for k,v in profile.items() if k.startswith('lab8_')}},
         'data':{'num_workers':0,'split_path':'splits/cifar10.json','course_protocol_status':'UNVERIFIED','cal_read':False},
         'train':{'amp':False},'runtime':{'device':'cuda','threads':2,'dependency_lock':'requirements.lock.txt'},'logging':{'run_root':'runs/lab8'},
         'ece':{'M':15,'binning':'equal_width'},'corruptions':PARAMETERS,'corruption_protocol':'local explicitly parameterized approximations; not certified CIFAR-10-C equivalence',
         'attacks':{'FGSM_eps_255':[0,1,2,4,8],'PGD_eps_255':profile['lab8_pgd_eps_255'],'K':10,'R':1,'alpha_rule':'2.5*epsilon/K','boundary_K':20,'boundary_R':2,'test_count':10000,'batch_size':64},
         'model_selection':{'fraction':.05,'optimizer_updates':10000,'checkpoint':'last.pt','reason':'Only this fraction has three genuine matched seeds. No new classifier training; no test-based checkpoint selection.'}}
    Path('configs/lab8.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    (R/'exploratory_hypotheses.md').write_text('''# Предварительные гипотезы E1 и E3 (seed 1)
E1: по (8.6) положительная температура сохраняет классы и точность. Минимизация NLL на чистой val не повышает её NLL, поскольку T=1 включена. Ожидаем снижение ECE на test, но это не гарантируется; порядок CNN/ViT по точности может не совпасть с ECE. Проверка M=5,15,50 и равного числа объектов выявит зависимость от измерителя.
E3: по (8.9)-(8.10) ожидаем accuracy(PGD10) <= accuracy(FGSM) при epsilon=2/255; PGD20 R2 может дополнительно уменьшить точность. До контроля Q3 предсказываем accuracy(random sign Gaussian) > accuracy(FGSM) >= accuracy(PGD10), так как случайное направление не максимизирует потерю. При несовпадении порядок сохраняется как наблюдение, а не переписывается.
Обе архитектуры: fraction=.05, 10000 обновлений в ЛР2, seed=1, одинаковые 10000 test ID, eval, исходные пиксели [0,1], нормализация первым слоем, фиксированные checkpoint last.pt. Температура обучается только на val. Никаких обновлений классификатора.
''',encoding='utf-8')
    print({k:v for k,v in profile.items() if k.startswith('lab8_')})

if __name__=='__main__':main()
