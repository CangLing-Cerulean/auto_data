"""Report new second-origin runs alongside explicitly reused first-origin evidence."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'research_runs/support_confirmation'
LABEL = {'balanced': '状态平衡采样', 'reshuffle': '不放回洗牌'}


def main():
    progress = json.loads((FOLDER / 'progress.json').read_text())
    assert not progress['phase'].startswith('running_')
    reused = []
    baseline = json.loads((ROOT / 'research_runs/results/T000_prefix60_full/result.json').read_text())['future']['groups']['all']
    for name, policy in zip(progress['reused_prefix60_runs'], ['balanced', 'reshuffle']):
        result = json.loads((ROOT / 'research_runs/results' / name / 'result.json').read_text())
        metrics = result['future']['groups']['all']
        reused.append({'run_id': name, 'origin': .6, 'policy': policy, 'seed': 2026, 'reused': True, **metrics,
            'NMSE_reduction_pct': 100 * (1 - metrics['NMSE'] / baseline['NMSE']),
            'NMAE_reduction_pct': 100 * (1 - metrics['NMAE'] / baseline['NMAE'])})
    profiles = []
    for row in progress['rows']:
        folder = ROOT / 'research_runs/results' / row['run_id']
        with np.load(folder / 'windows.npz') as z:
            counts = z['draw_count']; n = len(counts)
            assert counts.sum() == 768000
            profile = {'run_id': row['run_id'], 'eligible_windows': n, 'unique_windows_presented': int((counts > 0).sum()),
                'total_presentations': int(counts.sum()), 'minimum_presentations': int(counts.min()), 'maximum_presentations': int(counts.max())}
            if row['policy'] == 'balanced':
                q = z['q']; assert q.min() > 0 and abs(q.sum() - 1) < 1e-12
                with np.load(folder / 'coverage_policy.npz') as saved:
                    assert np.array_equal(q, saved['probability']) and np.array_equal(z['train_start'], saved['start_row'])
                profile.update(minimum_relative_probability=float(q.min() * n), maximum_relative_probability=float(q.max() * n))
            else:
                with np.load(folder / 'sampling_plan.npz') as saved:
                    assert np.array_equal(counts, np.bincount(saved['candidate_indices'].ravel(), minlength=n))
            profiles.append(profile)
    used = progress['budget_used_seconds']
    summary = {**progress, 'reused_prefix60_results': reused, 'sampling_profiles': profiles,
        'stage_used_seconds': used - progress['stage_start_seconds'],
        'combined_round_used_seconds': used - progress['round_start_seconds'], 'remaining_total_seconds': 7200 - used,
        'test_accessed': False}
    lines = ['# 状态覆盖与不放回洗牌：补充时间原点结果', '',
        '本阶段保持此前固定配方，补齐两条已有首折正信号、当时因排名未进入第二原点的规则。首折直接引用T023/T042，没有重跑或计作本轮新增；新实验及推进门槛见[PROTOCOL.md](PROTOCOL.md)。同轮此前的近期输入分布加权已停止，详见[对应报告](../covariate_sampling/REPORT.md)。', '',
        '## 结果', '',
        '| 证据来源/评价范围 | 方法 | NMSE | NMAE | NMSE降低 | NMAE降低 |',
        '|---|---|---:|---:|---:|---:|']
    for r in reused + progress['rows']:
        scope = '复用旧首折：前60%→随后20%' if r.get('reused') else '本轮新增：前80%→随后20%' if r['origin'] < 1 else f"本轮完整训练/原验证，seed{r['seed']}"
        lines.append(f"| {scope} | {LABEL[r['policy']]} | {r['NMSE']:.8f} | {r['NMAE']:.8f} | {r['NMSE_reduction_pct']:.4f}% | {r['NMAE_reduction_pct']:.4f}% |")
    lines += ['', '降低为负表示退化；每行只与相同原点/种子的uniform配对。内部绝对分数与正式验证不能混排。首折既有结果是开发选择依据，不是本轮新增独立重复。', '',
        f"终态：{progress['phase']}。进入正式验证的候选：{progress['winner']}。采用新默认：{progress['adopt_default']}。"]
    if 'mean_NMSE' in progress:
        lines += ['', f"正式三种子平均NMSE={progress['mean_NMSE']:.8f}、NMAE={progress['mean_NMAE']:.8f}，相对uniform分别降低{progress['mean_NMSE_reduction_pct']:.4f}%和{progress['mean_NMAE_reduction_pct']:.4f}%。"]
    lines += ['', '## 固定配方、覆盖与计费', '',
        '状态平衡直接调用原coverage_sampling.build函数，维持一半均匀、一半状态均衡的概率及原CDF随机流；不放回洗牌保持原NumPy种子与逐轮排列算法。原DLinear、训练标准化、数据值、窗口边界、MSE/Adam和训练更新数均不变。此阶段没有辅助拟合或新的评分前向。', '',
        '| 新运行 | 候选数 | 实际出现过的候选 | 单窗口最少/最多呈现 |',
        '|---|---:|---:|---:|']
    for p in profiles:
        lines.append(f"| {p['run_id']} | {p['eligible_windows']} | {p['unique_windows_presented']} | {p['minimum_presentations']}/{p['maximum_presentations']} |")
    lines += ['', f"本阶段完成{len(progress['rows'])}次新主训练，计费{summary['stage_used_seconds']:.4f}秒；与输入分布阶段合计{summary['combined_round_used_seconds']:.4f}秒，低于共享1200秒上限。原7200秒预算累计{used:.4f}秒，剩{7200 - used:.4f}秒，没有增加额度。", '',
        '所有新增运行均保存模型/优化器、源行、实际抽样次数，状态平衡保存固定q，不放回方案保存完整768000次抽样计划。计数及概率/计划对应关系核对通过。没有读取测试分布或得分，未触发ID均不排队。']
    if not progress['adopt_default']:
        lines += ['', '默认仍全部保留、均匀采样。本轮完成后根据上述实际结果判断这两条配方，不再沿用“第二原点尚未验证”的旧表述，也不扫描混合比例或随机种子以挑值。']
    (FOLDER / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    (FOLDER / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'phase': progress['phase'], 'rows': progress['rows'], 'combined_round_seconds': summary['combined_round_used_seconds'], 'total_seconds': used}))


if __name__ == '__main__': main()
