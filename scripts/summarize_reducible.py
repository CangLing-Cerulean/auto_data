"""Summarize reference-gap selection from saved results and sampling traces."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'research_runs/reducible_sampling'
LABEL = {'rho': '固定参考误差差值', 'adarho': '自适应参考误差差值',
         'hard_rank': '同候选池困难排名', 'shuffle': '胜者逐批分层位置打乱'}


def main():
    progress = json.loads((FOLDER / 'progress.json').read_text())
    assert not progress['phase'].startswith('running_')
    used = progress['budget_used_seconds']; profiles = []; auxiliary = []
    for row in progress['rows']:
        folder = ROOT / 'research_runs/results' / row['run_id']
        result = json.loads((folder / 'result.json').read_text())
        baseline = json.loads((ROOT / 'research_runs/results' / row['baseline'] / 'result.json').read_text())
        baseline_losses = [x.get('training_loss', x.get('train_loss')) for x in baseline['history'][:2]]
        actual_losses = [x['training_loss'] for x in result['history'][:2]]
        warmup = actual_losses == baseline_losses
        warmup_validation = None if row['origin'] < 1 else [x['validation'] for x in result['history'][:2]] == [x['validation'] for x in baseline['history'][:2]]
        with np.load(folder / 'windows.npz') as z:
            counts = z['draw_count']; proposed = z['proposed_count']; reference = z['reference_mask']; cells = z['cell']
            assert counts.sum() == 768000 and np.all(counts <= proposed)
            cell_profile = [{'cell': int(cell), 'candidate_population_share': float((cells == cell).mean()),
                             'actual_training_share': float(counts[cells == cell].sum() / counts.sum())} for cell in np.unique(cells)]
            profiles.append({'run_id': row['run_id'], 'warmup_training_matches': warmup, 'warmup_validation_matches': warmup_validation,
                'eligible_windows': len(counts), 'ever_selected_windows': int((counts > 0).sum()),
                'total_presentations': int(counts.sum()), 'proposed_presentations': int(proposed.sum()),
                'reference_day_population_share': float(reference.mean()),
                'reference_day_actual_training_share': float(counts[reference].sum() / counts.sum()),
                'reference_update_presentations': result['additional_reference_updates'],
                'selection_forward_presentations': result['selection_forward_presentations'],
                'selection_reference_forward_presentations': result['selection_reference_forward_presentations'],
                'cell_profile': cell_profile})
    for name in progress['auxiliary_runs']:
        folder = ROOT / 'research_runs/results' / name
        audit = json.loads((folder / 'data_audit.json').read_text())
        status = json.loads((folder / 'status.json').read_text())
        auxiliary.append({'run_id': name, 'origin': audit['origin'], 'eligible_windows': audit['reference_eligible_windows'],
                          'eligible_fraction': audit['reference_eligible_fraction'], 'elapsed_seconds': status['elapsed_seconds'],
                          'training_presentations': 768000})
    comparisons = []
    for origin in (.6, .8):
        rows = [r for r in progress['rows'] if r['origin'] == origin]
        selected = next((r for r in rows if r['policy'] == progress['winner']), None)
        control = next((r for r in rows if r['policy'] == 'shuffle'), None)
        if selected and control:
            comparisons.append({'origin': origin, 'winner_vs_shuffle_NMSE_reduction_pct': 100 * (1 - selected['NMSE'] / control['NMSE'])})
    summary = {**progress, 'sampling_profiles': profiles, 'auxiliary_profiles': auxiliary, 'comparisons': comparisons,
        'round_used_seconds': used - 4789.922976038419, 'remaining_total_seconds': 7200 - used,
        'completed_main_models': len(progress['rows']), 'completed_auxiliary_pretraining_models': len(auxiliary),
        'test_accessed': False}
    lines = ['# 文献驱动的参考误差差值与批内筛选', '',
        '本轮由用户再次授权继续探索并检索现有成果。基于[RHO-LOSS](https://proceedings.mlr.press/v162/mindermann22a.html)与[FAF/AdaRho](https://proceedings.mlr.press/v300/taga26a.html)，研究参考误差差值是否能改善训练窗口选择。作者实现与本项目适配边界见[RESEARCH.md](RESEARCH.md)，固定流程见[PROTOCOL.md](PROTOCOL.md)。', '',
        '主预测模型仍为原DLinear，仅改变训练窗口选择。辅助参考只在当前可用训练历史的整日子集训练；没有模型集成、修改传感器值、数据增强或测试反馈。所有主模型3000步、每步256窗口；候选池每步512个均匀窗口，1000步预热后开始排名。', '',
        '## 实际结果', '', '| 训练历史/评价范围 | 方法 | NMSE | NMAE | NMSE降低 | NMAE降低 |',
        '|---|---|---:|---:|---:|---:|']
    for r in progress['rows']:
        scope = f"前{r['origin'] * 100:.0f}% / 随后20%训练内部段" if r['origin'] < 1 else f"完整训练 / 原验证，seed{r['seed']}"
        lines.append(f"| {scope} | {LABEL[r['policy']]} | {r['NMSE']:.8f} | {r['NMAE']:.8f} | {r['NMSE_reduction_pct']:.4f}% | {r['NMAE_reduction_pct']:.4f}% |")
    lines += ['', '降低为负表示退化；内部与正式验证的绝对数值不能混排。首折统一seed2026，不能当作多种子结论。', '',
              f"首折胜者：{progress['winner']}。流程终态：{progress['phase']}。", '',
              f"跨时段及控制共同支持：{progress.get('internal_signal_confirmed', False)}。采用新默认：{progress['adopt_default']}。"]
    for c in comparisons:
        lines.append(f"前{c['origin'] * 100:.0f}%历史下，胜者相对其逐批分层打乱控制的NMSE降低{c['winner_vs_shuffle_NMSE_reduction_pct']:.4f}%。")
    if 'mean_NMSE' in progress:
        lines += ['', f"正式三种子均值NMSE={progress['mean_NMSE']:.8f}、NMAE={progress['mean_NMAE']:.8f}；分别相对uniform降低{progress['mean_NMSE_reduction_pct']:.4f}%和{progress['mean_NMAE_reduction_pct']:.4f}%。"]
    lines += ['', '## 代价、可追溯性与限制', '',
        f"完成{len(progress['rows'])}次主模型训练、{len(auxiliary)}次辅助参考预训练。新研究进程共{summary['round_used_seconds']:.4f}秒，低于本轮1800秒上限；原7200秒累计{used:.4f}秒，剩余{7200 - used:.4f}秒。未增加预算。", '',
        '每个参考预训练额外呈现768000个窗口；每个参考差值主运行在预热后额外计算1024000次主模型及1024000次参考模型评分前向；AdaRho另有256000次参考反向呈现。困难排名只计算主模型评分，打乱对照重放已保存选择。这些开销均已计入墙钟，不能宣称总计算量相同或已经证明训练加速。', '',
        '每个运行的windows.npz记录源start_row、训练/候选/参考更新计数、参考日成员关系和状态×月份；phase_*.npz保存逐步候选索引、选中位置及真实评分。动态选择没有一张固定概率表，未抽中的窗口不等于从基础数据删除。逐阶段模型、优化器、随机状态、源码和协议快照完整保留。', '',
        '参考模型预训练成员与主模型候选有交集；参考误差不等于已证明不可约噪声。批内上界2/N是条件期望训练占比的上界，不是有限抽样中每个窗口实际出现次数的硬上界，也没有宣称每个窗口至少保留0.5/N概率。', '',
        '前1000步训练损失与对应uniform基线完全一致：' + str(all(p['warmup_training_matches'] for p in profiles)) + '。完整验证预热一致性单独见summary.json。本轮只检查必要边界与保存的计数，没有重跑旧GPU审计。']
    if not progress['adopt_default']:
        lines += ['', '默认继续全量均匀采样。所有未触发的后续ID都不是待运行队列；下一轮先读真实结果再登记不同假设，不重置额度或扫本轮参考比例、学习率和候选池大小。']
    (FOLDER / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    (FOLDER / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'phase': progress['phase'], 'winner': progress['winner'], 'rows': progress['rows'],
        'main_models': len(progress['rows']), 'auxiliary_models': len(auxiliary), 'used': used,
        'warmup_match': all(p['warmup_training_matches'] for p in profiles)}))


if __name__ == '__main__': main()
