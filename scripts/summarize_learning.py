"""Summarize completed adaptive sampling experiments using saved metrics only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'research_runs/learning_sampling'
LABEL={'progress':'最近500步误差下降','hard':'当前误差较大','shuffle':'胜者概率分层打乱'}


def main():
    progress=json.loads((FOLDER/'progress.json').read_text())
    assert not progress['phase'].startswith('running_')
    rows=progress['rows']; used=progress['budget_used_seconds']
    wave_used=sum(json.loads((ROOT/'research_runs/results'/r['run_id']/'status.json').read_text())['elapsed_seconds'] for r in rows)
    profiles=[]
    for r in rows:
        folder=ROOT/'research_runs/results'/r['run_id']
        result=json.loads((folder/'result.json').read_text())
        baseline=json.loads((ROOT/'research_runs/results'/r['baseline']/'result.json').read_text())
        audit=json.loads((folder/'sampling_audit.json').read_text())
        warmup_equal=result['history'][:2]==baseline['history'][:2]
        profiles.append({'run_id':r['run_id'],'warmup_history_matches_baseline':warmup_equal,**audit})
    summary={**progress,'profiles':profiles,'wave_used_seconds':wave_used,'shared_round_used_seconds':used-4114.524273328483,'remaining_total_seconds':7200-used,'default_policy':'uniform','formal_validation_started':False,'test_accessed':False}
    lines=['# 训练反馈采样探索','','本轮比较两种训练窗口价值假设：提高当前误差大窗口的机会，或提高最近500步误差下降窗口的机会。保留全部候选，至少一半采样质量均匀分配；固定状态×月份的总质量，只在单元内调整。前1000步均匀，之后每500步更新。输入、标准化、7920参数DLinear、损失及3000次更新不变。','','## 实际结果','','训练历史 | 方法 | NMSE | NMAE | NMSE降低 | NMAE降低','--- | --- | ---: | ---: | ---: | ---:']
    for r in rows:
        lines.append(f"前{r['origin']*100:.0f}% | {LABEL[r['policy']]} | {r['NMSE']:.8f} | {r['NMAE']:.8f} | {r['NMSE_reduction_pct']:.4f}% | {r['NMAE_reduction_pct']:.4f}%")
    lines+=['','指标均在随后20%的训练内部时段计算；降低为负表示退化。不同时间原点的绝对数值不能直接混排。',f"首折胜者：{progress['winner']}。流程终态：{progress['phase']}。两折均改善且胜过打乱对照：{progress.get('internal_signal_confirmed',False)}。"]
    comparisons=[]
    for origin in (.6,.8):
        same=[r for r in rows if r['origin']==origin]
        target=next((r for r in same if r['policy']==progress['winner']),None)
        control=next((r for r in same if r['policy']=='shuffle'),None)
        if target and control:
            delta=100*(1-target['NMSE']/control['NMSE'])
            comparisons.append({'origin':origin,'winner_vs_shuffle_NMSE_reduction_pct':delta})
            lines.append(f"前{origin*100:.0f}%训练时，胜者相对同层打乱对照的NMSE降低{delta:.4f}%。")
    summary['comparisons']=comparisons
    lines+=['','## 证据边界与接续','','本轮仅训练内部开发反馈，未开展完整验证或测试，也未采用新默认。打乱对照使用胜者冻结的逐阶段概率，在相同状态×月份内置换，保留全部概率值和单元总质量；对照不利用自身模型重新评分。','', '评分方案额外执行五遍候选窗口前向，计入原墙钟预算。对照复用冻结分数，因此只有反向更新数和训练呈现次数相同，不能称总计算量完全相同。每阶段源窗口概率、训练误差、实际次数、评分模型与可续训模型均已保存。']
    if progress.get('internal_signal_confirmed'):
        lines+=['','下一步只对已冻结胜者登记完整验证，报告NMAE代价，随后按新登记规则决定种子确认；当前不能宣称正式优于uniform。']
    else:
        lines+=['','当前结果未建立该训练反馈规则的跨时段优势。停止此配方，不继续扫混合比例或评分频率；下一步应登记不同的数据价值假设，默认继续全部保留、均匀采样。']
    lines+=['',f"第二波完成运行用时{wave_used:.4f}秒；两波累计{summary['shared_round_used_seconds']:.4f}秒（共享上限1200秒）。原7200秒已用{used:.4f}秒，剩{7200-used:.4f}秒。",'', '前1000步训练损失与基线一致性：'+str(all(p['warmup_history_matches_baseline'] for p in profiles))+'。这是已有数值记录的轻量检查，未重跑历史GPU审计。']
    (FOLDER/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (FOLDER/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'phase':progress['phase'],'winner':progress['winner'],'rows':rows,'budget_used_seconds':used,'warmup_matches':all(p['warmup_history_matches_baseline'] for p in profiles)}))


if __name__=='__main__': main()
