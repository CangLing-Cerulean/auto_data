"""Summarize exploratory outcomes without repeated GPU audit or model fitting."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'research_runs/coverage_sampling'
LABEL={'balanced':'状态平衡','recent':'组内近期','balanced_recent':'状态平衡+组内近期','recent_shuffle':'组内近期位置打乱（对照）'}


def main():
    progress=json.loads((FOLDER/'progress.json').read_text())
    assert not progress['phase'].startswith('running_')
    used=progress['budget_used_seconds']
    rows=progress['rows']; summary={**progress,'budget_used_seconds':used,'round_used_seconds':used-4114.524273328483,'remaining_total_seconds':7200-used}
    lines=['# 全支持采样：状态覆盖与近期窗口探索','','## 研究目标与边界','','用户要求提高探索比重。本轮比较三个新的抽样策略和一个打乱对照，减少重复审计。保留全部合法窗口，只改变抽样概率；原始传感器表示、训练标准化、DLinear7920参数、Adam3000步、768000次呈现和MSE损失不变。测试未访问。','','一半概率质量来自均匀分布，另一半为状态平衡、组内30天半衰期近期权重，或两者结合。近期位置打乱保留组内概率值分布，破坏权重与时间位置的对应。分组来自A013前60%训练输入的冻结结果，不重新拟合。','','## 实际结果','','阶段 | 策略 | 种子 | NMSE | NMAE | NMSE降低 | NMAE降低','--- | --- | ---: | ---: | ---: | ---: | ---:']
    for r in rows:
        lines.append(f"{r['stage']}%训练{'/完整验证' if r['stage']==100 else '/后20%内部评估'} | {LABEL[r['policy']]} | {r['seed']} | {r['NMSE']:.8f} | {r['NMAE']:.8f} | {r['NMSE_reduction_pct']:.4f}% | {r['NMAE_reduction_pct']:.4f}%")
    lines+=['','降低为负数表示退化。内部时段的绝对分数不与完整验证混排。探索推进只按主指标NMSE的正向信号排名，不回改历史实验的停止门槛；NMAE代价完整保留。']
    if 'mean_NMSE' in progress:
        lines+=['',f"冻结胜者：{LABEL[progress['selected_policy']]}。三种子平均NMSE={progress['mean_NMSE']:.8f}，相对原uniform平均降低{progress['mean_NMSE_reduction_pct']:.4f}%；平均NMAE={progress['mean_NMAE']:.8f}，降低{progress['mean_NMAE_reduction_pct']:.4f}%。"]
    lines+=['',f"流程终态：{progress['phase']}。可采用新默认：{progress.get('adopt_default',False)}。",'', '若未达到三种子平均NMSE改善至少1%、每种子NMSE均改善、平均NMAE不退化，默认仍为原uniform；有主指标信号也只作为带限制的开发证据。']
    summary['sampling_profiles']=[]
    for r in rows:
        run=ROOT/'research_runs/results'/r['run_id']; audit=json.loads((run/'coverage_audit.json').read_text())
        summary['sampling_profiles'].append({'run_id':r['run_id'],'retained_windows':audit['retained_windows'],'min_relative_to_uniform':audit['min_relative_to_uniform'],'max_relative_to_uniform':audit['max_relative_to_uniform'],'groups':audit['groups'],'months':audit['months']})
    formal=[r for r in rows if r['stage']==100 and r['policy']!='recent_shuffle']
    if formal:
        details=[]
        for r in formal:
            value=json.loads((ROOT/'research_runs/results'/r['run_id']/'result.json').read_text())
            base=json.loads((ROOT/'research_runs/results'/r['baseline']/'result.json').read_text())
            details.append({'run_id':r['run_id'],'group_NMSE_reduction_pct':{k:100*(1-value['groups'][k]['NMSE']/base['groups'][k]['NMSE']) for k in base['groups']},'channel_NMSE_reduction_pct':{c:100*(1-a/b) for c,a,b in zip(value['sensor_columns'],value['per_channel_NMSE'],base['per_channel_NMSE'])}})
        summary['formal_details']=details
    lines+=['','## 已保留资产与必要检查','','各运行coverage_policy.npz保留源start_row、冻结组编号、真实input_end时间与概率；windows.npz或sampling_weights.npz保留实际抽样次数。best/final/latest检查点、优化器、随机状态、代码和协议快照均保留。运行中检查原数据哈希、窗口边界、概率正且归一、全部支持以及768000次实际呈现；本轮不额外重跑历史模型、聚类或全量随机流审计。','',f"第一波研究进程用时{summary['round_used_seconds']:.4f}秒；第一波结束时原7200秒累计已用{used:.4f}秒，剩余{7200-used:.4f}秒。第二波共享同一个1200秒探索额度，最新总账见state.json；预算未重置。"]
    (FOLDER/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (FOLDER/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'phase':progress['phase'],'rows':rows,'adopt_default':progress.get('adopt_default',False),'round_used_seconds':summary['round_used_seconds']}))


if __name__=='__main__': main()
