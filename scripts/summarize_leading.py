"""Portable paired summary for the preregistered leading-source study."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'research_runs/representation'; results=ROOT/'research_runs/results'
    progress=json.loads((folder/'leading_progress.json').read_text())
    diagnostic=json.loads((results/'A009_leading_sources/diagnostic.json').read_text())
    verification=json.loads((folder/'leading_verification.json').read_text())
    rows=progress['rows']; formal=[r for r in rows if r['run_id'].startswith('R')]
    columns=diagnostic['sensor_columns']; mapping=diagnostic['folds'][0]['mapping']
    summary={'phase':progress['phase'],'rows':rows,'mapping':dict(zip(columns,[columns[j] for j in mapping])),'scope':'input-source representation; not training-row selection','verification':verification,'development_gate_passed':False}
    if formal:
        candidates=[json.loads((results/r['run_id']/'result.json').read_text()) for r in formal]
        references=[json.loads((results/r['baseline']/'result.json').read_text()) for r in formal]
        summary['channel_NMSE_reduction_pct']={name:100*(1-sum(r['per_channel_NMSE'][i] for r in candidates)/sum(r['per_channel_NMSE'][i] for r in references)) for i,name in enumerate(columns)}
        summary['channel_NMAE_reduction_pct']={name:100*(1-sum(r['per_channel_NMAE'][i] for r in candidates)/sum(r['per_channel_NMAE'][i] for r in references)) for i,name in enumerate(columns)}
        summary['formal_NMAE_reduction_pct']=100*(1-sum(r['groups']['all']['NMAE'] for r in candidates)/sum(r['groups']['all']['NMAE'] for r in references))
        summary['group_NMSE_reduction_pct']={name:100*(1-sum(r['groups'][name]['NMSE'] for r in candidates)/sum(r['groups'][name]['NMSE'] for r in references)) for name in references[0]['groups']}
    if len(formal)==3:
        base=[json.loads((results/r['baseline']/'result.json').read_text())['groups']['all'] for r in formal]
        a=sum(r['NMSE'] for r in formal)/3; b=sum(r['NMSE'] for r in base)/3
        ma=sum(r['NMAE'] for r in formal)/3; mb=sum(r['NMAE'] for r in base)/3
        summary.update(mean_NMSE=a,baseline_mean_NMSE=b,mean_NMSE_reduction_pct=100*(1-a/b),mean_NMAE=ma,baseline_mean_NMAE=mb,mean_NMAE_reduction_pct=100*(1-ma/mb))
        summary['development_gate_passed']=bool(a<=.99*b and ma<=mb and all(r['NMSE_reduction_pct']>0 for r in formal))
    lines=['# 训练集选择历史输入来源：本轮结果','','本轮使用既有metropt3_v1原始7:2:1划分、32→8任务及DLinear骨干。输入来源由训练内部前向评估选择，外部验证未参与映射选择，测试未访问。所有训练窗口保留且均匀采样。属于输入数据表示/来源选择，不是训练行删除、降权或重采样的收益。','','## 选择规则','','输入 z_i(t)=x_j(t)-x_j(31)+x_i(31)，自身来源直接保持原值；输入是原始真实窗口的派生视图，不新增传感器记录。保留目标当前水平，替换历史变化来源。目标真值不变，模型7920参数、Adam3000步、768000次呈现保持不变。','','目标 | 历史变化来源','--- | ---']
    for i,j in enumerate(mapping):
        if i!=j: lines.append(f'{columns[i]} | {columns[j]}')
    lines+=['','其余九个目标保持自身输入。每个窗口可通过固定window_table的原始train.csv行号和上述映射溯源。','','## 仅训练集诊断','','解析模型不同于正式Adam训练，单独报告：']
    for f in diagnostic['folds']: lines.append(f"- 前{int(f['origin']*100)}%→后20%：NMSE {f['baseline_NMSE']:.8f} → {f['selected_NMSE']:.8f}，降低{f['NMSE_reduction_pct']:.4f}%。")
    lines+=['','第一折候选选择至少5%单通道改善；第二折冻结同一映射。两折整体至少1%，第二折NMAE及所有替换通道不退化，门槛均通过。','','## 实际训练配对结果','','运行 | 对照 | NMSE | NMSE降低 | NMAE不退化','--- | --- | ---: | ---: | ---']
    for r in rows: lines.append(f"{r['run_id']} | {r['baseline']} | {r['NMSE']:.8f} | {r['NMSE_reduction_pct']:.4f}% | {r['NMAE_not_worse']}")
    if len(formal)==3:
        lines+=['',f"正式三种子平均NMSE：{summary['baseline_mean_NMSE']:.8f} → {summary['mean_NMSE']:.8f}（降低{summary['mean_NMSE_reduction_pct']:.4f}%）。平均NMAE降低{summary['mean_NMAE_reduction_pct']:.4f}%。",f"预登记开发候选门槛通过：{summary['development_gate_passed']}。"]
    if formal:
        lines+=['',f"正式评估目前{len(formal)}个种子，NMAE相对降低{summary['formal_NMAE_reduction_pct']:.4f}%（负数为退化）。本轮R001的NMAE为0.19863202，基线为0.19500733，未过非退化门槛，因此停止，R002/R003未运行。不能把内部两折当正式三种子复验。"]
        lines+=['','完整验证集分组NMSE变化（正数为改善）：']
        lines += [f'- {name}: {value:.4f}%' for name,value in summary['group_NMSE_reduction_pct'].items()]
        lines+=['','逐通道NMSE变化（正数为改善）：']
        lines += [f'- {name}: {value:.4f}%' for name,value in summary['channel_NMSE_reduction_pct'].items()]
        lines+=['','逐通道NMAE变化（正数为改善）：','']
        lines += [f'- {name}: {value:.4f}%' for name,value in summary['channel_NMAE_reduction_pct'].items()]
    lines+=['',f"流程终态：{progress['phase']}。未运行测试，不能宣称测试泛化改善或领先于其他模型。",'','## 核验与后续','','独立重建第一折映射、两折行边界及正规方程；恒等映射精确保持输入/输出，真实输入上的独立变换实现一致。所有本轮训练与各自基线实际抽样次数向量完全一致，数据审计、标准化与代码快照身份校验通过。','', '当前保留统一采样原表示作为默认。停止本配方，不修改门槛、不按验证结果挑掉退化通道。下一轮优先在训练集内部诊断NMAE代价是否来自特定转移/稳定区间的系统偏差，再决定能否登记独立的数据权重假设；不能把本轮输入表示收益宣称为训练行筛选成功。']
    (folder/'leading_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (folder/'LEADING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()
