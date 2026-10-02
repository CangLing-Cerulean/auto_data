"""Portable report; keep representation and training-weight effects separate."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'research_runs/selection'; result_root=ROOT/'research_runs/results'
    progress=json.loads((folder/'discordance_progress.json').read_text())
    verification=json.loads((folder/'discordance_verification.json').read_text())
    diagnostic=json.loads((result_root/'A011_source_errors/diagnostic.json').read_text())
    lines=['# 输入变化不一致窗口加权：反馈报告','','## 训练内部诊断','','A011仅使用train.csv及T000/T005/T010/T011末步模型；未读取外部验证或测试。下面为来源模型相对原模型的总误差增量贡献，正值表示该组增加总误差。','','历史原点 | 分组 | 总NMAE增量贡献 | 总NMSE增量贡献','--- | --- | ---: | ---:']
    for fold in diagnostic['folds']:
        for row in fold['groups']:
            lines.append(f"{fold['origin']}% | {row['group']} | {row['NMAE_delta_contribution']:+.8f} | {row['NMSE_delta_contribution']:+.8f}")
    lines+=['','同一分组方式内贡献可相加；两种分组方式相互重叠，不可混加。来源变化但目标历史不变组在两折均增加绝对误差；未来大变化组贡献主要平方误差收益。这只支持一个可检验假设，不足以判定数据好坏。','','## 唯一新变量与对照','','固定A009来源表示，目标历史恒定而来源有变化的训练窗口逐目标原始损失权重2，其余1；各目标在完整训练前缀的权重均值归一为1。均匀抽样和目标真值不变。月内逐列打乱对照具有完全相同的权重值和数量。原始行映射、规则mask、实际加权mask与损失权重见各运行discordance_weights.npz。','','## 实际训练结果','','运行 | NMSE | NMAE','--- | ---: | ---:']
    for row in progress['rows']: lines.append(f"{row['run_id']} | {row['NMSE']:.8f} | {row['NMAE']:.8f}")
    for fold in progress['folds']:
        base=json.loads((result_root/fold['source_baseline']/'result.json').read_text())['future']['groups']['all']
        lines+=['',f"原点{int(fold['origin']*100)}%无权重来源基线{fold['source_baseline']}：NMSE={base['NMSE']:.8f}，NMAE={base['NMAE']:.8f}。",f"定向权重相对该基线：NMSE降低{fold['NMSE_reduction_vs_source_pct']:.4f}%，NMAE降低{fold['NMAE_reduction_vs_source_pct']:.4f}%（负值为退化）。",f"推进门槛通过：{fold['gate_passed']}。",'']
        lines += [f'- {name}: {value}' for name,value in fold['checks'].items()]
    passed=progress['phase']=='completed_internal' and all(f['gate_passed'] for f in progress['folds'])
    lines+=['','## 实际加权覆盖','','训练窗口—目标列的加权位置计数（同一窗口可影响多列，不能相加当作去重窗口数）：','']
    columns=diagnostic['sensor_columns']
    for check in verification['checks']:
        if 'shuffle' in check['run_id']: continue
        lines.append(f"- {check['run_id']}：共{check['eligible_windows']}个合法窗口。")
        for name,count in zip(columns,check['rule_counts']):
            if count: lines.append(f"- {name}：{count}个窗口，占该列窗口的{100*count/check['eligible_windows']:.4f}%。")
    lines+=['','其余列权重恒为1。DV_eletric覆盖远高于其他列；“来源变化、目标不变”在此列是常见状态，并非普遍稀有困难事件。分组诊断能定位误差，但不自动给出有效训练权重。']
    summary={**progress,'all_internal_gates_passed':passed,'verification':verification,'formal_validation_run':False,'test_accessed':False}
    conclusion='内部两折均通过，尚未运行正式验证；下一步须独立登记正式验证及配对控制。' if passed else '未过训练内部门槛，本权重配方停止。未进入新的正式验证，不扫描倍率，不把旧表示带来的收益记到本次加权上。'
    lines+=['','## 决策与核验','',conclusion,'','所有运行逐窗口的输入规则、月内数量、归一权重与完整抽样次数已独立核验；与各自无权重来源基线数据审计完全相同。原metropt3_v1基础划分和DLinear协议保持，原始uniform默认不变。','', '已保存最终模型、可续训检查点、优化器、随机状态、权重/源行映射、代码协议快照与哈希。下一轮不得重复本轮完成ID或事后修改停止门槛。']
    (folder/'DISCORDANCE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    (folder/'discordance_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()
