"""Summarize the frozen covariate-sampling study without model evaluation."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'research_runs/covariate_sampling'
LABEL = {'global': '全局近期密度比', 'state_matched': '状态质量固定的近期密度比', 'shuffle': '胜者概率在状态×月份内打乱'}


def main():
    progress = json.loads((FOLDER / 'progress.json').read_text())
    assert not progress['phase'].startswith('running_')
    auxiliary = []; profiles = []
    for name in progress['auxiliary_runs']:
        result = json.loads((ROOT / 'research_runs/results' / name / 'result.json').read_text())
        auxiliary.append({key: result[key] for key in ['run_id', 'eligible_windows', 'recent_windows', 'recent_population_share',
            'latest_input_end', 'recent_start', 'fitted_rows_per_domain', 'source_recent_fit_overlap',
            'in_sample_domain_accuracy', 'in_sample_domain_BCE', 'optimizer_closure_calls', 'profiles', 'elapsed_seconds']})
    for row in progress['rows']:
        run = ROOT / 'research_runs/results' / row['run_id']
        result = json.loads((run / 'result.json').read_text())
        with np.load(run / 'windows.npz') as z:
            q = z['q']; counts = z['draw_count']; starts = z['train_start']; cells = z['cell']
            assert counts.sum() == 768000 and abs(q.sum() - 1) < 1e-12
            assert q.min() * len(q) >= .5 - 1e-9 and q.max() * len(q) <= 2 + 1e-9
            with np.load(ROOT / 'research_runs/results' / result['identity']['score_run'] / 'density_weights.npz') as score:
                assert np.array_equal(starts, score['train_start'])
                recent = score['recent']
            profiles.append({'run_id': row['run_id'], 'eligible_windows': len(q), 'total_presentations': int(counts.sum()),
                'unique_windows_presented': int((counts > 0).sum()), 'minimum_relative_probability': float(q.min() * len(q)),
                'maximum_relative_probability': float(q.max() * len(q)), 'recent_probability_mass': float(q[recent].sum()),
                'recent_actual_training_share': float(counts[recent].sum() / counts.sum()),
                'cell_profile': [{'cell': int(c), 'probability_mass': float(q[cells == c].sum()),
                    'actual_training_share': float(counts[cells == c].sum() / counts.sum())} for c in np.unique(cells)]})
    comparisons = []
    for origin in (.6, .8):
        rows = [r for r in progress['rows'] if r['origin'] == origin]
        selected = next((r for r in rows if r['policy'] == progress['winner']), None)
        shuffled = next((r for r in rows if r['policy'] == 'shuffle'), None)
        if selected and shuffled:
            comparisons.append({'origin': origin, 'selected_vs_shuffle_NMSE_reduction_pct': 100 * (1 - selected['NMSE'] / shuffled['NMSE']),
                'selected_vs_shuffle_NMAE_reduction_pct': 100 * (1 - selected['NMAE'] / shuffled['NMAE'])})
            selected_profile = next(p for p in profiles if p['run_id'] == selected['run_id'])
            shuffled_profile = next(p for p in profiles if p['run_id'] == shuffled['run_id'])
            for a, b in zip(selected_profile['cell_profile'], shuffled_profile['cell_profile']):
                assert a['cell'] == b['cell'] and abs(a['probability_mass'] - b['probability_mass']) < 1e-12
    used = progress['budget_used_seconds']
    summary = {**progress, 'auxiliary_profiles': auxiliary, 'sampling_profiles': profiles, 'comparisons': comparisons,
        'round_used_seconds': used - progress['round_start_seconds'], 'remaining_total_seconds': 7200 - used,
        'completed_main_models': len(progress['rows']), 'completed_auxiliary_domain_models': len(auxiliary), 'test_accessed': False}
    lines = ['# 近期输入分布相似度采样研究', '',
        '本轮测试“与训练历史最近30个自然日输入工况更相似的历史窗口是否更有价值”。密度比思路来自[NeurIPS 2023研究](https://papers.nips.cc/paper_files/paper/2023/file/5cad96c4433955a2e76749ee74a424f5-Paper-Conference.pdf)；这里使用训练侧近期代理，没有读取随后评价段输入或任何测试数据来拟合权重，未复现原论文在线算法。预登记规则见[PROTOCOL.md](PROTOCOL.md)。', '',
        '原DLinear、传感器表示、标准化、3000步、batch256及768000次主训练呈现不变。辅助逻辑回归只读取32条输入的均值、标准差和首尾差；每个窗口概率始终为均匀的0.5—2倍，所有候选保留。全局规则允许工况质量变化，state_matched固定每个既有工况的总质量。', '',
        '## 真实预测结果', '',
        '| 训练历史/评价范围 | 方法 | NMSE | NMAE | NMSE降低 | NMAE降低 |',
        '|---|---|---:|---:|---:|---:|']
    for r in progress['rows']:
        scope = f"前{r['origin'] * 100:.0f}% / 随后20%训练内部段" if r['origin'] < 1 else f"完整训练 / 原验证，seed{r['seed']}"
        lines.append(f"| {scope} | {LABEL[r['policy']]} | {r['NMSE']:.8f} | {r['NMAE']:.8f} | {r['NMSE_reduction_pct']:.4f}% | {r['NMAE_reduction_pct']:.4f}% |")
    lines += ['', '降低为负表示退化；各行相对同时间原点/种子的uniform。不同时间原点与正式验证绝对分数不能混排。', '',
        f"首折候选中最低NMSE规则：{progress['winner']}。终态：{progress['phase']}。采用新默认：{progress['adopt_default']}。"]
    for c in comparisons:
        lines.append(f"前{c['origin'] * 100:.0f}%原点，入选规则相对分层位置打乱对照的NMSE降低{c['selected_vs_shuffle_NMSE_reduction_pct']:.4f}%，NMAE降低{c['selected_vs_shuffle_NMAE_reduction_pct']:.4f}%。")
    if 'mean_NMSE' in progress:
        lines += ['', f"正式三种子均值NMSE={progress['mean_NMSE']:.8f}、NMAE={progress['mean_NMAE']:.8f}，相对uniform降低{progress['mean_NMSE_reduction_pct']:.4f}%/{progress['mean_NMAE_reduction_pct']:.4f}%。"]
    lines += ['', '## 输入评分器与采样行为', '',
        '| 辅助运行 | 近期候选占比 | 判别器训练准确率 | 源/近期拟合成员交集 |',
        '|---|---:|---:|---:|']
    for a in auxiliary:
        lines.append(f"| {a['run_id']} | {100 * a['recent_population_share']:.3f}% | {100 * a['in_sample_domain_accuracy']:.3f}% | {a['source_recent_fit_overlap']} |")
    lines += ['', '准确率是辅助判别器在自身拟合样本上的描述，不能作为预测泛化或校准密度比的证据。源域为全部可用历史，近期代理是其子集，成员交集已记录。未来分布可能变化，条件输出分布不变也未被证明。', '',
        '| 主运行 | 候选数 | 实际出现过 | 近期概率质量 | 近期实际呈现占比 |',
        '|---|---:|---:|---:|---:|']
    for p in profiles:
        lines.append(f"| {p['run_id']} | {p['eligible_windows']} | {p['unique_windows_presented']} | {100 * p['recent_probability_mass']:.3f}% | {100 * p['recent_actual_training_share']:.3f}% |")
    lines += ['', '打乱控制保留各状态×月份的概率总质量和概率值集合，比较的是层内具体窗口位置；同层也会保留部分时间/状态信息，不能把它理解为所有结构完全随机。源行、评分模型、概率、实际计数、预测模型、优化器与随机状态完整保存。', '',
        '## 预算和停止', '',
        f"完成{len(progress['rows'])}次主预测训练和{len(auxiliary)}次辅助分布判别器拟合，本轮计费{summary['round_used_seconds']:.4f}秒，低于1200秒上限。原7200秒预算累计{used:.4f}秒，剩{7200 - used:.4f}秒。所有辅助开销计入预算，没有声称总计算量与uniform相同。", '',
        '未触发的后续ID没有运行或排队。没有修改基础数据、验证真值或读取测试分布/得分；不把输入分布可区分性等同于训练窗口价值。']
    if not progress['adopt_default']:
        lines += ['', '默认继续保留全部合法窗口并均匀采样。本轮规则没有达到采用条件，不继续扫描近期天数、特征组合、正则强度或权重界限。']
    (FOLDER / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    (FOLDER / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'phase': progress['phase'], 'rows': progress['rows'], 'auxiliary_models': len(auxiliary),
        'round_seconds': summary['round_used_seconds'], 'used_seconds': used}))


if __name__ == '__main__': main()
