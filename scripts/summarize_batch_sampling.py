"""Report batch-organization experiments from frozen results and draw plans."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'research_runs/batch_sampling'
LABELS = {'reshuffle': '全候选不放回洗牌', 'time_stratified': '均匀CDF时间分层组批'}


def main():
    progress = json.loads((FOLDER / 'progress.json').read_text())
    assert not progress['phase'].startswith('running_')
    profiles = []
    for row in progress['rows']:
        run = ROOT / 'research_runs/results' / row['run_id']
        result = json.loads((run / 'result.json').read_text())
        with np.load(run / 'windows.npz') as z:
            counts = z['draw_count']; n = len(counts)
            assert counts.sum() == 768000
            with np.load(run / 'sampling_plan.npz') as plan:
                assert np.array_equal(counts, np.bincount(plan['candidate_indices'].ravel(), minlength=n))
            profiles.append({'run_id': row['run_id'], 'eligible_windows': n,
                'unique_windows_presented': int((counts > 0).sum()), 'unique_fraction': float((counts > 0).mean()),
                'minimum_presentations': int(counts.min()), 'maximum_presentations': int(counts.max()),
                'total_presentations': int(counts.sum()), 'best_step': result.get('best_step')})
    used = progress['budget_used_seconds']
    summary = {**progress, 'sampling_profiles': profiles,
        'stage_used_seconds': used - progress['stage_start_seconds'],
        'combined_exploration_used_seconds': used - 4789.922976038419,
        'remaining_total_seconds': 7200 - used, 'test_accessed': False}
    lines = ['# 均匀目标下的批次组织研究', '',
        '参考误差差值和困难排名未改善首个训练内部原点后，转向不改变长期均匀目标的批次组织。依据与预登记推进规则见[PROTOCOL.md](PROTOCOL.md)。借鉴[Random Reshuffling](https://proceedings.neurips.cc/paper_files/paper/2020/hash/c8cc6e90ccbff44c9cee23611711cdc4-Abstract.html)的不放回训练和[SCott](https://proceedings.mlr.press/v139/lu21d.html)的时序分层思想；原Adam不变，未复现SCott优化器。', '',
        '两种规则均从第一步启用，全部合法训练窗口保留，原DLinear、标准化、3000步、batch256和768000次训练呈现不变。无辅助模型或额外模型评分前向。', '',
        '## 结果', '',
        '| 训练历史/评价范围 | 方法 | NMSE | NMAE | NMSE降低 | NMAE降低 |',
        '|---|---|---:|---:|---:|---:|']
    for r in progress['rows']:
        scope = f"前{r['origin'] * 100:.0f}% / 随后20%训练内部段" if r['origin'] < 1 else f"完整训练 / 原验证，seed{r['seed']}"
        lines.append(f"| {scope} | {LABELS[r['policy']]} | {r['NMSE']:.8f} | {r['NMAE']:.8f} | {r['NMSE_reduction_pct']:.4f}% | {r['NMAE_reduction_pct']:.4f}% |")
    lines += ['', '降低为负表示退化；不同原点和正式验证的绝对数值不能混排。训练内部均以最终3000步检查点评价；正式验证按原每500步评价并选择最低NMSE检查点。', '',
        f"首折胜者：{progress['winner']}。流程终态：{progress['phase']}。跨两个内部原点信号：{progress.get('internal_signal_confirmed', False)}。采用新默认：{progress['adopt_default']}。"]
    if 'mean_NMSE' in progress:
        lines += ['', f"三种子正式均值NMSE={progress['mean_NMSE']:.8f}、NMAE={progress['mean_NMAE']:.8f}；相对uniform分别降低{progress['mean_NMSE_reduction_pct']:.4f}%和{progress['mean_NMAE_reduction_pct']:.4f}%。"]
    lines += ['', '## 实际覆盖与成本', '',
        '| 运行 | 合法候选 | 实际出现过 | 覆盖比例 | 单窗口最少/最多呈现 |',
        '|---|---:|---:|---:|---:|']
    for r in profiles:
        lines.append(f"| {r['run_id']} | {r['eligible_windows']} | {r['unique_windows_presented']} | {100 * r['unique_fraction']:.4f}% | {r['minimum_presentations']}/{r['maximum_presentations']} |")
    lines += ['', '覆盖比例反映本次有限768000次呈现，未被抽中的窗口仍属于保留的候选数据。洗牌每轮无重复；时间分层针对按源行排序的合法窗口的等概率CDF段，每步各段一次独立随机抽取。后者控制批内时间组成，不把窗口数量解释为固定持续时间，也不改变窗口内部顺序。', '',
        f"此阶段完成{len(progress['rows'])}次主模型训练，用时{summary['stage_used_seconds']:.4f}秒。与参考差值阶段合计用时{summary['combined_exploration_used_seconds']:.4f}秒，低于共享1800秒上限。原7200秒预算累计{used:.4f}秒，余{7200 - used:.4f}秒，没有重置预算。", '',
        '各运行保存完整抽样计划及SHA256、实际计数、源start_row、源码/协议快照、模型与优化器；计数已与完整计划核对。没有访问测试分布或得分。未触发的后续ID没有运行，也不构成待运行队列。']
    if not progress['adopt_default']:
        lines += ['', '默认继续全部保留、均匀采样。首折的小幅改善不能作为采用证据；后续应基于真实跨原点和正式验证反馈登记新假设，不扫描分层数或重排种子以挑最好结果。']
    (FOLDER / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    (FOLDER / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'phase': progress['phase'], 'rows': progress['rows'], 'stage_seconds': summary['stage_used_seconds'], 'total_seconds': used}))


if __name__ == '__main__': main()
