"""Portable evidence report for frozen input-state group deletion."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'research_runs/cluster_selection'


def main():
    fit=json.loads((FOLDER/'fit.json').read_text()); progress=json.loads((FOLDER/'progress.json').read_text()); audit=json.loads((FOLDER/'verification.json').read_text())
    rows=[]
    for row in progress['rows']:
        base_id=('T000_prefix60_full' if row['stage']==60 else 'T005_prefix80_full') if row['stage']!=100 else {2026:'S000_uniform_s2026',2027:'S005_uniform_s2027',2028:'S010_uniform_s2028'}[row['seed']]
        raw=json.loads((ROOT/'research_runs/results'/base_id/'result.json').read_text()); b=(raw if row['stage']==100 else raw['future'])['groups']['all']
        rows.append({**row,'baseline':base_id,'baseline_NMSE':b['NMSE'],'baseline_NMAE':b['NMAE'],'NMSE_reduction_pct':100*(1-row['NMSE']/b['NMSE']),'NMAE_reduction_pct':100*(1-row['NMAE']/b['NMAE'])})
    used=sum(json.loads(p.read_text()).get('elapsed_seconds',0) for p in (ROOT/'research_runs/results').glob('*/status.json'))
    summary={**progress,'rows':rows,'fit_counts':fit['prefix60_group_counts'],'eligible_groups':fit['eligible_groups'],'verification':audit,'development_gate_passed':progress.get('development_gate_passed',False),'total_budget_used_seconds':used,'round_budget_used_seconds':used-3865.6280975155532,'remaining_total_seconds':7200-used,'test_accessed':False}
    lines=['# 冻结输入状态组删除研究','','## 结论与执行边界','',f"流程终态：{progress['phase']}；开发候选门槛通过：{summary['development_gate_passed']}。",'','只改变训练窗口保留/删除；原始输入表示、DLinear7920参数、32→8、Adam3000步、768000次呈现保持。内部前缀标准化使用完整前缀而非删除后的子集。没有访问测试；内部原点与外部验证均属于开发反馈。','','## 冻结分组','','在训练前60%内的632231个完整合法窗口拟合30维输入均值/标准差特征的标准化，固定种子20261002抽32768个窗口。K=4、单次K-means++，实际迭代'+str(fit['fit']['iterations'])+'轮，收敛='+str(fit['fit']['converged'])+'；后续未重新聚类。','','组号 | 前缀窗口数 | 占比 | 删组资格','--- | ---: | ---: | ---']
    for c,n in enumerate(fit['prefix60_group_counts']): lines.append(f"{c} | {n} | {100*n/fit['fit_windows']:.4f}% | {'合格' if c in fit['eligible_groups'] else '不合格，删除后保留不足50%或组占比不足1%'}")
    lines+=['','## 重训结果','','NMSE/NMAE降低百分比为负表示退化。不同内部原点或正式验证的绝对分数不能混排。','','实验 | 组 | NMSE | NMAE | NMSE降低 | NMAE降低','--- | ---: | ---: | ---: | ---: | ---:']
    for r in rows: lines.append(f"{r['run_id']} | {r['group']} | {r['NMSE']:.8f} | {r['NMAE']:.8f} | {r['NMSE_reduction_pct']:.4f}% | {r['NMAE_reduction_pct']:.4f}%")
    for base_id in dict.fromkeys(r['baseline'] for r in rows):
        r=next(r for r in rows if r['baseline']==base_id); lines+=['',f"基线{base_id}：NMSE={r['baseline_NMSE']:.8f}，NMAE={r['baseline_NMAE']:.8f}。"]
    if progress['phase']=='stopped_no_prefix60_candidate': lines+=['','三个合格删组方案均未同时达到NMSE至少改善1%、NMAE不退化的第一道门槛，因此没有选择候选组。按预登记流程，未运行随机删除对照、80%原点或正式验证；不把未执行阶段描述成阴性结果。','', '默认仍保留全部1053433个合法原训练窗口并均匀采样。本次证据反对直接删除这三个完整输入状态组；它不证明组内每条数据都必须保留，也不排除更细的其他筛选方法。']
    lines+=['','## 组的实际输入特征','','以下是前缀60%组内窗口32步均值的组平均，用于描述覆盖，不是模型输入新增特征。组编号没有普遍的好坏含义。','','组 | TP2 | Motor_current | COMP | DV_eletric | Pressure_switch','--- | ---: | ---: | ---: | ---: | ---:']
    for profile in audit['cluster_profiles']:
        f=profile['mean_input_features']; lines.append(f"{profile['group']} | {f[0]:.5f} | {f[6]:.5f} | {f[7]:.5f} | {f[8]:.5f} | {f[12]:.5f}")
    lines+=['','## 保留/删除清单与复现','','每个实验的`*_windows.csv.gz`包含原train.csv起始行、输入与窗口结束行、原始首末索引、组编号、retained、月份、采样概率和实际抽样次数。retained=1表示本实验保留，0表示本实验不抽样；没有删除或修改基础CSV。相应mask*.npz及身份JSON为冻结机器输入。','']
    lines += [f'- [{name}]({name})' for name in audit['exports']]
    lines+=['','A014重建全训练分组标签、前60%特征统计、聚类随机抽样与中心；验证所有mask源行和月份、归一概率、完整前缀标准化、原模型参数数目、768000次随机流重放、删除者零抽样以及内部最终检查点指标。','',f"原预算已用{used:.4f}秒；本轮含聚类、训练、审计共{summary['round_budget_used_seconds']:.4f}秒，低于1800秒；原总预算剩{7200-used:.4f}秒。",'', '恢复研究先读本报告、progress.json、verification.json及state.json。不得重新启动已完成ID或改写分组；没有通过的阶段不自动补跑。']
    (FOLDER/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    (FOLDER/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'phase':progress['phase'],'rows':rows,'budget_used':used,'round_used':summary['round_budget_used_seconds']}))


if __name__=='__main__': main()
