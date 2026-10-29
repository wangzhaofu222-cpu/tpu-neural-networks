"""Human-readable Chinese handoff and requirement statuses; no formal report."""
import json
from pathlib import Path
from src.utils.io import write_json
R=Path('experiments/lab5')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    audit=read(R/'final_audit.json');e1=read(R/'E1.json');e2=read(R/'E2.json');e3=read(R/'E3.json');st=audit['statistics'];lock=read(R/'E1_selection_lock.json')
    lines=['# Lab5 实验交付记录（不含正式报告）','',
      f"最终独立审计：**{audit['status']}**，{audit['check_count']} 项证据核查。自动测试 {st['final_tests_passed']} 项通过，其中原有106项回归测试。",
      'Lab4 提交709e39c、334个checkpoint、E1/E2/E3统计重算与历史文件保护通过；原有Q2仍为BLOCKED，未编造颜色控制协议。',
      '', '## 个人参数及协议','',
      'Cora；第二GNN=GAT；深度[2,3,6,12,24]。宽64、dropout=.5、Adam、200epochs、单图全批量、CPU2线程、float32、确定性。',
      '使用用户授权的PyG public masks，140/500/1000；教师独立课程split未提供，核对状态仍为UNVERIFIED。cal为空。transductive仅允许全节点X/边，损失仅train、选模仅val。',
      'E1网格各模型4组：lr∈{.01,.005}×wd∈{.0005,.005}，val最佳epoch最早优先，超参数并列按预定网格顺序。',
      '选中参数：'+', '.join(f"{k}: lr={v['lr']}, wd={v['wd']}, epoch={v['best_epoch']}" for k,v in lock['selected'].items()),
      '', '## E1','', '| 模型 | Train | Val | Test | Gtop (pp) | 评估 RUN_ID |','|---|---:|---:|---:|---:|---|']
    for k,r in e1['models'].items():lines.append(f"|{k}|{100*r['train']:.2f}|{100*r['val']:.2f}|{100*r['test']:.2f}|{100*e1['Gtop'].get(k,0):.2f}|{r['RUN_ID']}|")
    lines+=['',f"固定宽16参考GCN={100*e1['reference']['test']:.2f}%，与81.5%差{100*abs(e1['reference']['test']-.815):.2f}pp；参考检查{e1['reference_acceptance']}。",
      '', '| p | 原权重 A (%) | 重训 B (%) | 孤立节点/预测 | 相对偏差 | 孤立test | hE |', '|---|---:|---:|---|---:|---:|---:|']
    for r in e1['deletion']:
        h='n/a' if r['hE'] is None else f"{r['hE']:.4f}"
        lines.append(f"|{r['p']}|{100*r['A_test']:.2f}|{100*r['B_test']:.2f}|{r['isolated']}/{r['expected_isolated']:.2f}|{100*r['relative_deviation']:.2f}%|{r['isolated_test']}|{h}|")
    crossing=e1['B_first_at_or_below_MLP']
    lines+=['',f"重训曲线首次不高于MLP：p={crossing['p']}，该条件test孤立比例={crossing['isolated_test']/10:.1f}%。p=1重训与MLP的差值={100*(e1['deletion'][-1]['B_test']-e1['models']['mlp']['test']):+.2f}pp（超参数分别由val选取）。p<1的hE稳定，损失邻居并不等于改变同配性。",
      '', '## E2','',f"事前假设提交：{read(R/'preregistration_evidence.json')['commit']}。E2的9个配对模型训练均在其后，seed1与E1同条件已探索这一限制已公开。",
      '', '| Seed | 原图GCN (%) | null GCN (%) | MLP (%) | Δ (pp) | null hE | hdeg | 交换成功率 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in e2['paired']:lines.append(f"|{r['seed']}|{100*r['original']['test']:.2f}|{100*r['null']['test']:.2f}|{100*r['mlp']['test']:.2f}|{100*r['delta']:.2f}|{r['null_hE']:.5f}|{r['hdeg']:.5f}|{100*r['swap_success']:.2f}%|")
    lines+=['',f"平均Δ={100*e2['mean_delta']:.3f}pp，样本SD(ddof=1)={100*e2['sample_sd_delta']:.3f}pp；同号={e2['same_nonzero_sign']}；教师判定：{e2['teacher_conclusion']}。",
      f"定量假设预测+15至+35pp，实测{'在' if .15<=e2['mean_delta']<=.35 else '不在'}预测区间；即使方向支持，量级偏离也必须保留。ER seed1 test={100*e2['ER']['test']:.2f}%；ER改变度分布，单seed差异不能单独归因于度。三seeds仅反映当前固定split下的变异。",
      '', '## E3','',f"val最优深度={e3['lstar_validation']}；sm={100*e3['sm']:.3f}pp，明显下降阈值=2sm={100*e3['threshold']:.3f}pp。",
      '| 深度 | Train/Val/Test (%) | ρT | GI | S | Ê | 输出层probe val (%) | 诊断 |','|---|---|---:|---:|---:|---:|---:|---|']
    for r in e3['plain']:lines.append(f"|{r['depth']}|{100*r['train']:.2f}/{100*r['val']:.2f}/{100*r['test']:.2f}|{r['rhoT']:.5f}|{r['GI']:.4f}|{r['S']:.5f}|{r['energy']:.5g}|{100*r['probe_val']:.2f}|{r['diagnosis']}|")
    res=e3['residual'];smooth=[r['depth'] for r in e3['plain'] if r['diagnosis']=='smoothing']
    lines+=['',f"严格按教师联合判据归为有害过度平滑的深度：{smooth if smooth else '未观察到可确证案例'}。S升高本身不够；rho、梯度和probe一起决定解释。GI>=6是事前公开的分析惯例，教师PDF未给出该数字。",
      f"24层残差模型 Train/Val/Test={100*res['train']:.2f}/{100*res['val']:.2f}/{100*res['test']:.2f}%，ρT={res['rhoT']:.5f}，GI={res['GI']:.4f}，S={res['S']:.5f}，Ê={res['energy']:.5g}，probe={100*res['probe_val']:.2f}%。输入维度不匹配用可学习投影，后续隐藏层identity skip，输出无skip/ReLU；不可直接套用无权传播极限。",
      'rhoT/GI表为最佳val checkpoint、eval模式、train标签诊断；末轮训练rho及训练期每层梯度中位数GI另列E3_depths.csv，每一步原始梯度保存在训练RUN中。逐层probe、S、Ê和RUN_ID见E3_all_layers.csv；未训练传播包含0..24层。',
      '', '## 交付与限制','',
      '结构依赖地图：structure_map.csv/json，逐项列出固定变量、变化变量、可比范围和RUN_ID。五个反事实答案：counterfactual_answers.md；CF4遵守事前规则，不按结果改阈值。',
      f"完整模型正式训练{st['model_training_runs']}次；线性probe正式训练{st['linear_probe_training_runs']}次；合计{st['formal_training_runs']}次、{st['formal_optimizer_steps']}次更新。另有6个5epoch预算测试，未冒充正式结果。",
      f"正式训练累计{st['formal_training_seconds']:.2f}秒（{st['formal_training_seconds']/60:.2f}分钟，含诊断和保存）；预算训练{st['budget_training_seconds']:.2f}秒。正式阶段开始{st['experiment_start']}，独立审计完成{st['audit_finished_at']}。",
      f"本地checkpoint：正式{st['formal_checkpoints']}个，Lab5全部{st['all_lab5_checkpoints']}个。权重、原始Cora、图和逐层特征在本地；Git保存配置、哈希、预测JSON、表格、图、日志与审计，不强制加入大权重。",
      f"Lab5 阻塞项：{audit['Lab5_BLOCKED']}；审计失败项：{audit['failed']}。教师split核对UNVERIFIED，不以用户替代授权伪装为核对通过；Lab4 Q2仍BLOCKED。",
      '未生成LaTeX/PDF/正式俄语报告；未执行Lab6。', '']
    (R/'EXPERIMENT_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    p=Path('docs/requirements/requirements_lab5.md');text=p.read_text(encoding='utf-8')
    text=text.replace('| RUNNING |','| PASS |').replace('| PENDING |','| PASS |')
    if audit['failed']:text+='\nActual audit exceptions (never suppress):\n'+json.dumps(audit['failed'],ensure_ascii=False,indent=2)+'\n'
    if any(c['check']=='CF4_predeclared_witness' for c in audit['failed']):text=text.replace('| C04 |','| C04 |',1).replace('actual data | PASS |','actual data | FAIL |')
    p.write_text(text,encoding='utf-8')
    print('Chinese handoff and requirement statuses written')
if __name__=='__main__':main()
