"""Summarize completed cadence interventions and issue scoped receipts."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'research_runs/cadence_selection'


def digest(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    progress = read(FOLDER / 'progress.json')
    assert progress['phase'].startswith('stopped_') or progress['phase'] == 'completed_formal_confirmation'
    profiles = []; receipt = {}
    for row in progress['rows']:
        run = ROOT / 'research_runs/results' / row['run_id']
        assert read(run / 'status.json')['phase'] == 'completed'
        audit = read(run / 'cadence_audit.json')
        with np.load(run / 'windows.npz') as z, np.load(run / 'cadence_policy.npz') as saved:
            assert np.array_equal(z['train_start'], saved['train_start'])
            assert np.array_equal(z['probability'], saved['probability'])
            assert np.array_equal(z['loss_weight'], saved['loss_weight'])
            q = z['probability']; weights = z['loss_weight']; draws = z['draw_count']
            assert draws.sum() == 768000 and not np.any(draws[q == 0]) and abs(q.sum() - 1) < 1e-12
            assert np.isclose(q @ weights, audit['expected_loss_weight'])
            assert np.array_equal(saved['flagged'], saved['score'] > .1)
            cells = saved['cell']
            assert all(saved['flagged'][cells == c].sum() == saved['applied'][cells == c].sum() for c in np.unique(cells))
            baseline = ROOT / 'research_runs/results' / row['baseline']
            if audit['applied_policy'] == 'soft':
                with np.load(baseline / ('windows.npz' if row['origin'] < 1 else 'sampling_weights.npz')) as b:
                    assert np.array_equal(draws, b['draw_count'])
            profiles.append({'run_id': row['run_id'], 'policy': row['policy'], 'applied_policy': audit['applied_policy'],
                'origin': row['origin'], 'eligible_windows': audit['eligible_windows'], 'flagged_windows': audit['flagged_windows'],
                'flagged_fraction': audit['flagged_fraction'], 'retained_windows': audit['retained_windows'],
                'median_interval_seconds': audit['median_interval_seconds'], 'unique_windows_presented': int((draws > 0).sum()),
                'flagged_actual_presentations': int(draws[saved['flagged']].sum()),
                'applied_actual_presentations': int(draws[saved['applied']].sum()), 'sample_presentations': int(draws.sum()),
                'expected_loss_weight': audit['expected_loss_weight'], 'score_quantiles': audit['score_quantiles'],
                'cell_matching_verified': True, 'sampling_and_weights_verified': True})
        hashes = {p.name: digest(p) for p in run.iterdir() if p.is_file() and p.name != 'asset_checksums.json'}
        save(run / 'asset_checksums.json', hashes)
        for name, value in hashes.items(): receipt[(run / name).relative_to(ROOT).as_posix()] = value
    used = sum(read(p).get('elapsed_seconds', 0) for p in (ROOT / 'research_runs/results').glob('*/status.json'))
    summary = {**progress, 'profiles': profiles, 'budget_used_seconds': used,
        'round_used_seconds': used - progress['round_start_seconds'], 'remaining_seconds': 7200 - used,
        'no_new_test_evaluation': True}
    save(FOLDER / 'summary.json', summary)
    label = {'drop': '剔除高间隔偏差', 'soft': '高间隔偏差软降权', 'shuffle': '状态×月份匹配随机对照'}
    lines = ['# 采样间隔一致性的数据处理实验', '',
        '原DLinear、原传感器输入、标准化和训练更新预算固定，仅改变训练窗口保留或损失权重。预登记见[PROTOCOL.md](PROTOCOL.md)。MLP成果保持冻结，本轮没有测试集读取或新测试模型评估。', '',
        '时间戳规则仅在当前可用训练历史拟合间隔中位数m；40条训练窗口内39个间隔的RMS偏差/m>0.10标为高偏差。硬剔除移出本实验候选池；软降权设损失权重0.25并按全体期望归一。匹配对照在冻结状态×月份内重分配同数量处理标记。', '',
        '## 结果', '',
        '| 训练范围 / 评估范围 | 方法 | NMSE | NMAE | NMSE降低 | NMAE降低 |',
        '|---|---|---:|---:|---:|---:|']
    for row in progress['rows']:
        scope = f"前{int(row['origin']*100)}%训练→随后20%训练" if row['origin'] < 1 else f"完整训练→原验证 seed{row['seed']}"
        lines.append(f"| {scope} | {label[row['policy']]} | {row['NMSE']:.8f} | {row['NMAE']:.8f} | {row['NMSE_reduction_pct']:.4f}% | {row['NMAE_reduction_pct']:.4f}% |")
    lines += ['', '降低为负表示退化。内部结果仅与同原点T000/T005配对，正式结果与同种子S000/S005/S010配对；不同评估范围的绝对分数不能混排。', '',
        f"首阶段选中方式：{progress['winner']}。终态：{progress['phase']}。采用新数据处理默认：{progress['adopt_default']}。", '',
        '## 实际处理与控制', '',
        '| 运行 | 原合法候选 | 高偏差候选 | 保留候选 | 实际呈现 |',
        '|---|---:|---:|---:|---:|']
    for p in profiles:
        lines.append(f"| {p['run_id']} | {p['eligible_windows']} | {p['flagged_windows']} ({p['flagged_fraction']:.3%}) | {p['retained_windows']} | {p['sample_presentations']} |")
    lines += ['', '所有处理数量在状态×月份内与匹配控制一致；软降权与对应uniform的逐窗口抽样次数完全一致；零概率成员未被抽中。完整时间戳分数、标记、源行、概率、权重、次数、模型和源码已保存。', '',
        '较大的时间间隔偏差可能来自真实采样节拍变化，不能据此直接给记录贴错误/噪声标签。训练目标区间的时间戳参与训练样本选择，但不进入预测模型输入；全部评价窗口和真值原样保留。此控制不保证所有未观测工况完全一致。', '',
        f"本轮完成{len(progress['rows'])}次主训练，计费{summary['round_used_seconds']:.6f}秒；原7200秒累计{used:.6f}秒，剩{summary['remaining_seconds']:.6f}秒。未重置预算。", '',
        '未通过门槛的后续ID不构成待运行队列；停止后不扫描本规则阈值、损失倍率或控制种子。无新测试结论。']
    (FOLDER / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for path in FOLDER.iterdir():
        if path.is_file() and path.name != 'download_receipt.json': receipt[path.relative_to(ROOT).as_posix()] = digest(path)
    save(FOLDER / 'download_receipt.json', receipt)
    print(json.dumps({'phase': progress['phase'], 'rows': progress['rows'], 'profiles': profiles,
        'budget_used_seconds': used, 'round_used_seconds': summary['round_used_seconds'], 'receipt_files': len(receipt)}))


if __name__ == '__main__': main()
