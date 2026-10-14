"""Build retrospective tables and a static comparison figure from audited scalars."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args();out=args.stage
read=lambda name:json.loads((out/name).read_text(encoding='utf-8-sig'))
audit=read('final_audit.json');pred=read('predictions.json');assert audit['status']=='PASS'
methods=['M1','RAW','SOURCE','SHORT','FIXED'];colors=['#17796c','#517ca4','#b95b32','#9b83ab','#8d959d']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(13.8,4.7),layout='constrained')
series=[('Prediction MAE (percentage points)',audit['prediction_MAE_pp']),('Mean realized recommendation gain (%)',audit['mean_recommendation_gain']),('Batch cold cost (seconds)',audit['batch_cold_total_seconds'])]
for ax,(title,values) in zip(axes,series):
    ys=[values[m] for m in methods];bars=ax.bar(methods,ys,color=colors,width=.62)
    ax.axhline(0,color='#444',lw=.7);ax.set_title(title,fontsize=11,pad=13);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.bar_label(bars,labels=[f'{v:.2f}' for v in ys],padding=4,fontsize=9);ax.margins(y=.18)
fig.suptitle('Hydraulic intervention response: retrospective comparison',fontsize=15,fontweight='bold')
fig.supxlabel('12 task-action means / 36 seed-level responses. Outcomes were already known; costs include shared real parents.',fontsize=9)
fig.savefig(out/'comparison.png',dpi=190);fig.savefig(out/'comparison.svg');plt.close(fig)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
counts={m:sum(d['accepted_harm'] for d in audit['decisions'] if d['method']==m) for m in methods}
mae=audit['prediction_MAE_pp'];improvement=100*(mae['SOURCE']-mae['M1'])/mae['SOURCE']
max_error=max(r['max_absolute_delta_error'] for r in audit['checked']);max_ratio=max(r['max_error_bound_ratio'] for r in audit['checked'])
methodcost=sum(c['SOURCE'] for c in audit['costs']);mincost=min(c['SOURCE'] for c in audit['costs']);maxcost=max(c['SOURCE'] for c in audit['costs'])
protocol_sha=hashlib.sha256((out/'PROTOCOL.md').read_bytes()).hexdigest()
text=f'''# SOURCE公式适配与M1的Hydraulic回顾性比较

2026-10-14。完整36项SOURCE预测与独立审计通过。本轮在已揭示的G1/G2/G3全部四动作、三个原种子上补齐固定SOURCE公式适配：M1的12格平均响应误差为{mae['M1']:.6f}个百分点，SOURCE为{mae['SOURCE']:.6f}；M1三任务推荐的真实平均改善为{audit['mean_recommendation_gain']['M1']:.6f}%，SOURCE为{audit['mean_recommendation_gain']['SOURCE']:.6f}%。SOURCE速度显著更快，但两项推荐造成实际伤害。

这是已知结果上的事后基线补充，不能作为独立确认、跨数据泛化、严格等预算比较或作者Adam实现复现。M1在原MetroPT3/SKAB批次的停止边界不变，不据本轮结果恢复失败候选或调整SOURCE分段、P、阻尼、阈值。

## 同池结果

改善为正表示降低NMSE；预测误差单位为百分点。MAE先对每个任务动作的三种子预测/真值取均值，再比较全部12格。逐种子误差单独列于下表，不能与此主MAE混用。类别采用原±0.25%不变带；任务推荐只看三种子平均预测，最大值至少1%才执行。

{table(['方法','12格MAE','类别命中','推荐真实平均改善','伤害推荐','整批冷启动秒'],[[m,f"{mae[m]:.6f}",f"{round(audit['category_accuracy'][m]*12)}/12",f"{audit['mean_recommendation_gain'][m]:.6f}%",f"{counts[m]}/3",f"{audit['batch_cold_total_seconds'][m]:.6f}"] for m in methods])}

![五方法误差、决策收益与成本](comparison.png)

M1相对本SOURCE适配的MAE低{improvement:.4f}%，三个任务均不劣；推荐平均收益高{audit['mean_recommendation_gain']['M1']-audit['mean_recommendation_gain']['SOURCE']:.6f}个百分点，最差任务为0%。原20%/0.05个百分点误差尺度、2/3任务不劣和0.10个百分点推荐收益尺度均达到，但只作为事后描述，不补写新验证成功。

{table(['任务','M1 MAE','RAW MAE','SOURCE MAE','SHORT MAE','FIXED MAE'],[[tid]+[f"{audit['task_MAE_pp'][m][tid]:.6f}" for m in methods] for tid in ('G1','G2','G3')])}

{table(['种子','M1逐项MAE','RAW逐项MAE','SOURCE逐项MAE','SHORT逐项MAE','FIXED逐项MAE'],[[str(seed)]+[f"{audit['per_seed_MAE_pp'][m][str(seed)]:.6f}" for m in methods] for seed in (2026101701,2026101702,2026101703)])}

## 全部动作与决策

{table(['任务','动作','真实改善%','M1预测%','RAW预测%','SOURCE预测%','SHORT预测%','FIXED预测%'],[[c['task'],c['action'],f"{c['truth']:.6f}"]+[f"{c['predictions'][m]:.6f}" for m in methods] for c in audit['cells']])}

{table(['任务','方法','推荐','真实改善%','oracle遗憾百分点'],[[d['task'],d['method'],d['action'],f"{d['gain']:.6f}",f"{d['regret']:.6f}"] for d in audit['decisions']])}

SOURCE在G1将KEEP_FIRST预测为+{pred['recommendations']['G1']['means']['KEEP_FIRST']:.6f}%，实际−44.718922%；在G3将同动作预测为+{pred['recommendations']['G3']['means']['KEEP_FIRST']:.6f}%，实际−82.764085%。这两项是实际方向错误，不能以较低成本抵消。G2选中KEEP_SECOND，与M1相同，实际改善52.675287%。本轮仅能确认该固定适配在这些有限幅度操作上不足；不能仅凭这些标量区分分段/P假设、有限干预线性化及随机Adam路径误差的贡献。

## 计算与审计边界

SOURCE配置和两份公式模块保持工程阶段哈希：500步段、段末父θ/v、固定偏差校正Adam对角P、父完整二次曲率、标准归一化概率梯度差、机器精度负特征值截断。每段对δ做线性传播，参数读出为父θ+δ后转float32，再用原DLinear完整前向。没有optimizer.step、新完整训练或学习新超参数；这不是作者Adam实现复现。

独立审计通过13,824个非零响应块，增广矩阵指数重算的最大δ绝对误差{max_error:.12g}，最大误差/登记容限比{max_ratio:.12g}。容限按块为1e−7+1e−8×参考最大绝对值。432份候选B/C误差摘要逐字节一致，另核对108份父B/C前向摘要及B选点；15份父/动作A矩独立重建最大误差为{audit['max_moment_error']:.1f}。87个预测输入、旧审计真值、代码与全部新资产前后哈希一致。旧四方法的12格和主统计重新计算与原审计一致。

只使用已登记A/B/C和父状态，不访问D或基础validation/test，不新增传感器记录。完整训练累计373不变。公式数值正确不等于预测有效，SOURCE的负结果完整保留。

## 成本

SOURCE九个任务种子方法成本合计{methodcost:.6f}秒，每组{mincost:.6f}–{maxcost:.6f}秒，包括重新从A构建父+4动作矩（三种子共享并等分）、父状态读取、全部四动作传播/前向/保存/哈希。整批冷启动{pred['cold_total_seconds']:.6f}秒，另加了历史真实父训练、旧ABC/概率准备及当前载入。九组均低于历史60秒参考，但此处没有重新执行旧方法，不能宣称严格等预算胜出。

预测阶段实测墙钟{pred['phase_seconds']:.6f}秒，已分配执行时间{pred['booked_execution_seconds']:.6f}秒，未分配预检/汇总等开销{pred['other_phase_seconds']:.6f}秒。独立审计{audit['audit_seconds']:.6f}秒另记；Python启动、网络、最终批次JSON/计时收据写入不在这些计时内，也没有冒充零成本。旧方法沿用原实际冷启动成本及载入时间，因此硬件时段、缓存和实现差异仍在。

## 保存与下一步

科学源代码冻结提交：`{read('source_sync_receipt.json')['source_commit']}`；完整预测冻结提交：`{read('prediction_sync_receipt.json')['source_commit']}`。本协议SHA256：`{protocol_sha}`。最终GitHub与Release收据见同目录sync_receipt.json；大型参数/矩资产见archive_receipt.json和release_receipt.json，原历史检查点继续使用旧归档。

本轮补上了一个低成本、与训练动态相关的公式基线，未推翻M1在Hydraulic旧池上的相对结果。主要证据缺口仍是跨源泛化和M1的有效成本/可靠性边界。后续应在原停止边界内先归纳失败原因与可检验假设，另行冻结任何新研究；不据本轮事后优势直接新增标签或扩大方法主张。
'''
with (out/'REPORT.md').open('x',encoding='utf-8') as f:f.write(text)
print(json.dumps(dict(report='REPORT.md',figure='comparison.png',max_delta_error=max_error,max_error_bound_ratio=max_ratio,method_seconds=methodcost)))
