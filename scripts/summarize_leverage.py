"""Saved-result report and lightweight checks for leverage sampling."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'research_runs/leverage_sampling'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    progress = read(FOLDER / 'progress.json')
    assert progress['phase'].startswith('stopped_') or progress['phase'] == 'completed_formal_confirmation'
    diagnostics = []; profiles = []; receipt = {}
    for name in progress['diagnostic_runs']:
        run = ROOT / 'research_runs/results' / name
        result = read(run / 'result.json')
        assert result['model_fitted'] is False and result['future_target_values_used_in_score'] is False
        with np.load(run / 'leverage_weights.npz') as z, np.load(run / 'geometry.npz') as g:
            score = z['score']; n = len(score); q = z['probability']; relative = q * n
            assert abs(score.mean() - 1) < 1e-8 and abs(q.sum() - 1) < 1e-10
            interior = (relative > .500001) & (relative < 1.999999)
            assert interior.any()
            alpha = float(np.median(relative[interior] / score[interior]))
            assert np.allclose(relative, np.clip(alpha * score, .5, 2), atol=1e-10, rtol=1e-10)
            eye = np.eye(33)
            assert np.allclose((g['gram'] + g['ridge'][:, None, None] * eye) @ g['inverse'], eye, atol=1e-8)
            assert np.allclose(g['effective_df'], (g['gram'] * g['inverse'].transpose(0, 2, 1)).sum((1, 2)))
        diagnostics.append({'run_id': name, 'windows': n, 'score_mean': float(score.mean()),
            'score_quantiles': result['score_quantiles'], 'effective_df': result['effective_df'],
            'elapsed_seconds': read(run / 'status.json')['elapsed_seconds'], 'numerical_checks_passed': True})
    counts_by_origin_policy = {}
    for row in progress['rows']:
        run = ROOT / 'research_runs/results' / row['run_id']; result = read(run / 'result.json')
        score_run = ROOT / 'research_runs/results' / result['identity']['score_run']
        with np.load(run / 'windows.npz') as z, np.load(score_run / 'leverage_weights.npz') as source:
            n = len(z['train_start']); q = z['probability']; weights = z['loss_weight']; counts = z['draw_count']
            assert np.array_equal(z['train_start'], source['train_start']) and np.array_equal(z['cell'], source['cell'])
            assert counts.sum() == 768000 and abs(q.sum() - 1) < 1e-10
            assert q.min() * n >= .5 - 1e-9 and q.max() * n <= 2 + 1e-9
            if result['corrected']:
                assert np.allclose(q * weights, 1. / n, atol=1e-18, rtol=1e-12)
            else: assert np.array_equal(weights, np.ones(n))
            expected = source['probability'].copy()
            if row['policy'] == 'shuffle':
                rng = np.random.default_rng(20261004)
                for cell in np.unique(z['cell']):
                    indices = np.flatnonzero(z['cell'] == cell)
                    expected[indices] = expected[rng.permutation(indices)]
            expected /= expected.sum()
            assert np.array_equal(q, expected)
            counts_by_origin_policy[(row['origin'], row['policy'], row['seed'])] = counts.copy()
            profiles.append({'run_id': row['run_id'], 'windows': n, 'corrected': result['corrected'],
                'relative_probability_min': float(q.min() * n), 'relative_probability_max': float(q.max() * n),
                'loss_weight_min': float(weights.min()), 'loss_weight_max': float(weights.max()),
                'expected_loss_weight': float(q @ weights), 'unique_windows_presented': int((counts > 0).sum()),
                'draw_count_sum': int(counts.sum()), 'probability_and_correction_verified': True})
    assert np.array_equal(counts_by_origin_policy[(.6, 'uncorrected', 2026)], counts_by_origin_policy[(.6, 'corrected', 2026)])
    all_runs = progress['diagnostic_runs'] + [r['run_id'] for r in progress['rows']]
    for name in all_runs:
        run = ROOT / 'research_runs/results' / name; assert read(run / 'status.json')['phase'] == 'completed'
        hashes = {p.name: digest(p) for p in run.iterdir() if p.is_file() and p.name != 'asset_checksums.json'}
        save(run / 'asset_checksums.json', hashes)
        for name, value in hashes.items(): receipt[(run / name).relative_to(ROOT).as_posix()] = value
    used = sum(read(p).get('elapsed_seconds', 0) for p in (ROOT / 'research_runs/results').glob('*/status.json'))
    summary = {**progress, 'diagnostics': diagnostics, 'profiles': profiles, 'budget_used_seconds': used,
        'stage_used_seconds': used - progress['stage_start_seconds'],
        'combined_round_used_seconds': used - progress['round_start_seconds'], 'remaining_seconds': 7200 - used,
        'initial_corrected_uncorrected_draw_counts_identical': True, 'no_new_test_evaluation': True}
    save(FOLDER / 'summary.json', summary)
    labels = {'uncorrected': '杠杆采样，无校正', 'corrected': '杠杆采样，逆概率校正', 'shuffle': '状态×月份概率打乱控制'}
    lines = ['# 输入杠杆分数采样：数据处理研究结果', '',
        '固定原DLinear和原传感器输入，只改变训练窗口采样及配套损失权重。全部合法窗口保留。分数由当前训练输入的33维逐通道Gram矩阵拟合，不用预测目标值、当前模型损失、验证输入或测试反馈；MLP成果保持冻结。', '',
        '依据[线性回归杠杆采样的统计分析](https://proceedings.mlr.press/v32/ma14.html)与[多维时序杠杆采样研究](https://proceedings.mlr.press/v89/xie19a.html)登记本适配；它们不保证当前DLinear/Adam协议泛化改善。详细定义与停止规则见[PROTOCOL.md](PROTOCOL.md)。', '',
        '## 结果', '', '| 范围 | 方法 | NMSE | NMAE | NMSE降低 | NMAE降低 |', '|---|---|---:|---:|---:|---:|']
    for row in progress['rows']:
        scope = f"前{int(row['origin']*100)}%训练→随后20%" if row['origin'] < 1 else f"完整训练→原验证 seed{row['seed']}"
        lines.append(f"| {scope} | {labels[row['policy']]} | {row['NMSE']:.8f} | {row['NMAE']:.8f} | {row['NMSE_reduction_pct']:.4f}% | {row['NMAE_reduction_pct']:.4f}% |")
    lines += ['', '降低为负代表退化。各行与同原点或同种子uniform配对，内部绝对值不与完整验证混排。', '',
        f"选中版本：{progress['winner']}。终态：{progress['phase']}。采用新默认：{progress['adopt_default']}。", '',
        '## 实际边界与可解释性', '',
        '相对均匀的概率固定限制在[0.5,2]；校正版权重为1/(Nq)，不作批内自归一化。校正保持原始随机梯度的期望对应uniform训练目标，不能据此称Adam更新无偏或推断测试收益。无校正版真实改变经验目标，两者差异用于辨别这两类作用。', '',
        '首次两个版本的逐窗口实际抽样次数完全一致；打乱控制保持每个状态×月份内概率分布，并重算对应逆概率权重。实际概率、权重、呈现总数、分层置换、Gram逆矩阵及分数归一化检查通过。所有训练均3000步、768000次呈现，额外矩阵统计/评分另计诊断开销。', '',
        '| 分数拟合 | 历史窗口数 | 分数均值 | 诊断秒数 |', '|---|---:|---:|---:|']
    for row in diagnostics:
        lines.append(f"| {row['run_id']} | {row['windows']} | {row['score_mean']:.12f} | {row['elapsed_seconds']:.4f} |")
    lines += ['', f"本阶段完成{len(progress['rows'])}次主训练和{len(diagnostics)}次输入几何诊断，用时{summary['stage_used_seconds']:.6f}秒；与时间间隔阶段合计{summary['combined_round_used_seconds']:.6f}秒，共享1000秒上限。原7200秒累计{used:.6f}秒，剩{summary['remaining_seconds']:.6f}秒。", '',
        '没有新测试访问或评分。未触发ID不是待运行任务；不按本轮结果扫描评分岭系数、概率边界或随机种子。默认策略以本轮预登记确认条件决定，旧MLP的最终测试结论单独保留。']
    (FOLDER / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for path in FOLDER.iterdir():
        if path.is_file() and path.name != 'download_receipt.json': receipt[path.relative_to(ROOT).as_posix()] = digest(path)
    save(FOLDER / 'download_receipt.json', receipt)
    print(json.dumps({'phase': progress['phase'], 'rows': progress['rows'], 'diagnostic_runs': progress['diagnostic_runs'],
        'combined_round_seconds': summary['combined_round_used_seconds'], 'budget_used_seconds': used, 'receipt_files': len(receipt)}))


if __name__ == '__main__': main()
