"""Summarize the frozen gradient-only audit without reading model responses."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261014_sampling_weight_information_r1'
def read(name):
    return json.loads((STAGE / name).read_text(encoding='utf-8-sig'))

result, audit, summary = (read(n) for n in ('diagnostic.json', 'final_audit.json', 'summary.json'))
assert result['status'] == 'completed' and audit['status'] == 'PASS'
rows = [row for task in result['tasks'] for row in task['records']]
assert len(rows) == 450
pairs = [r for r in rows if not r['identity_control']]
assert len(pairs) == 360
table = ['| 任务 | 动作 | W/S 中位数 [最小, 最大] | S 方差迹中位数 | W 方差迹中位数 |',
         '|---|---|---:|---:|---:|']
for group in summary['groups']:
    selected = [r for r in pairs if r['task'] == group['task'] and r['action'] == group['action']]
    ratios = [r['weighting_over_sampling_variance'] for r in selected]
    assert len(selected) == 18 and statistics.median(ratios) == group['ratio_median']
    table.append(f"| {group['task']} | {group['action']} | {group['ratio_median']:.6f} [{min(ratios):.6f}, {max(ratios):.6f}] | {group['sampling_trace_median']:.6g} | {group['weighting_trace_median']:.6g} |")
text = f'''# 采样与损失加权：梯度信息边界诊断

2026-10-14。阶段已完成计算、独立审计及下载核验；发布状态以同目录同步收据为准。新增完整训练0、累计373，新增代理训练、真实子标签、测试评价均0。

## 本轮结论

相同 μ=pw 的两种实现，在既有三来源全部父锚点上具有相同的期望梯度，却存在不同的梯度方差。360个动作配对中343项 W 方差更大、17项更小；90项PARENT恒等控制全部一致。KEEP动作的W/S范围为1.999121–2.011984，接近两倍。G1 FAVOR_SECOND则在18锚点中的17项更小，中位数0.815051、范围0.647144–1.007201，说明损失加权并非一致增加方差。

这些是训练梯度的代数统计，尚未证明实际Adam更新方差、有限训练预测误差或推荐差异，也不解释旧M/S选择增量为零的因果机制。均值相同而噪声不同是已有理论事实，本轮只量化M1旧设计明确列出的信息缺口；无新方法成立结论。

## 口径与完整结果

S使用旧采样p及权重1；W均匀采样全部N个A窗口、损失乘Np，batch按256个加权损失算术平均，保持每窗口 μ 相同。全部四动作、三来源五任务、每任务三种子和六个父检查点事前登记，未按结果删选。M/S组大小不等时按实际Np计算。

下表单样本梯度协方差迹使用原66维参数坐标，batch256均值梯度迹等于表中值除256。每格为18个相关父锚点的描述性统计，不能当18个独立任务或用于独立样本显著性检验；三个G任务同属Hydraulic。各任务绝对数值受尺度影响。所有450行绝对量、二阶矩、batch量、权重与误差见 [all_records.csv](all_records.csv)，全部统计见Release归档。

{chr(10).join(table)}

## 独立验收与成本

主实现逐窗口累加梯度范数，独立实现以四阶充分统计重建二阶矩，并重建原参数映射、G/R与配对差异；450行完整性/唯一性/公式和180项自动微分检查全部通过。最大矩差异 {summary['max_moment_gap']:.3e}，最大期望梯度差异 {summary['max_expected_gradient_gap']:.3e}，恒等控制最大绝对方差差异 {summary['max_parent_variance_gap']:.3e}。输入115文件前后哈希一致。浮点特征按原float32分解，统计在float64计算，未冒充实际float32 Adam轨迹。

冻结源码提交 d441284ead3f8101ff5a1a2086bcbb9519d4e4f1 已在运行前核验。主计算 {summary['diagnostic_seconds']:.6f}秒，独立审计 {summary['audit_seconds']:.6f}秒；管理、传输和人工整理未独立计时，保持null。进程43472与43626已终止，完成文件齐全，无后续模型队列。归档4,121,967字节、28成员均核验，SHA256 `1faf1d2c27459c7e48030bf2d3fbaaea5a2ad8dfe0bd3127e5493bbdfd9f142d`。

## 整体位置、下一步及论文证据缺口

旧M1原数值失败保持；补验和Hydraulic筛选开发收益仍有效。SOURCE事后比较支持旧池相对结果，M/S批次因预算停止且同推荐使选择增量门槛不可达，未完成真实子标签。本轮没有重启这些关闭批次、改变旧预测/成本门槛或读取B/C数值、D及基础测试。Appliances继续暂停，fs/fs_public禁用。

本轮支持继续检验M1无法表达的p/w区别是否影响有限训练响应。下一步另冻Hydraulic全部36个已有采样动作的损失加权对照，复用原父与初始化规则、共同A/B/C及固定训练协议，预测和真实标签分离；所有新加权结果揭示前锁定36项及审计/报告口径。不按本轮方差高低筛选动作，不把S与W间响应差直接归因于方差迹一个标量；这将是结构性局限检验，尚无自动启动授权于本协议内。

论文仍缺：p/w信息缺口在真实有限训练中的实质影响、能够改善诊断/选择的新方法、独立来源确认以及冻结后的最终评估。当前统计证据不能替代这些环节。
'''
(STAGE / 'REPORT.md').write_text(text, encoding='utf-8')
print('Report written; 20 groups, 450 complete records, no model response access.')
