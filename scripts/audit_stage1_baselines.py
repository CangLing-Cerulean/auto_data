"""Audit fixed references and export construction-only facts; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    started = time.monotonic()
    protocol = read(root/'baseline_protocol.json')
    results = {}
    for task, action, seed in protocol['registered_runs']:
        name = f'{task}_{action}_s{seed}'
        folder = root/'baseline_runs'/name
        result = read(folder/'result.json')
        assert read(folder/'status.json')['status'] == 'completed'
        assert result['protocol_sha256'] == sha(root/'baseline_protocol.json')
        assert result['source_sha256'] == protocol['runner_sha256']
        assert result['prepared_sha256'] == protocol['task_array_sha256'][task]
        for asset, expected in result['assets'].items():
            assert sha(folder/asset) == expected
        with np.load(folder/'best_errors.npz', allow_pickle=False) as saved:
            error = saved['error'].astype(np.float64)
        assert np.isfinite(error).all()
        np.testing.assert_allclose([np.mean(error**2), np.mean(abs(error)), error.mean()],
                                   [result['metrics'][key] for key in ('NMSE', 'NMAE', 'signed_bias')], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.mean(error**2, axis=(0, 1)), result['metrics']['channel_NMSE'], rtol=1e-12)
        assert result['best_step'] == min(result['history'], key=lambda x: x['NMSE'])['step']
        with np.load(folder/'sampling.npz', allow_pickle=False) as saved:
            assert saved['draw_count'].sum() == 768000
            assert np.all(saved['loss_weight'] == 1.)
            assert np.all(saved['probability'] == 1/len(saved['probability']))
        results[task+'_'+action] = result
    paired = []
    for task in ('C1', 'C2'):
        b, f = results[task+'_B0'], results[task+'_FIX']
        for key in ('initialization_sha256', 'probability_sha256', 'draw_count_sha256'):
            assert b[key] == f[key]
        paired.append(dict(task=task, global_NMSE_gain_percent=100*(1-f['metrics']['NMSE']/b['metrics']['NMSE']),
                           global_NMAE_gain_percent=100*(1-f['metrics']['NMAE']/b['metrics']['NMAE'])))
    # Only construction tasks enter the proposal/signal-building state.
    tasks = []
    for task in ('C1', 'C2'):
        receipt = read(root/'tasks'/task/'receipt.json')
        with np.load(root/'tasks'/task/'arrays.npz', allow_pickle=False) as arrays:
            values = arrays['train'].astype(np.float64)
            starts = arrays['train_starts']
            sampled = starts[np.linspace(0, len(starts)-1, min(8192, len(starts))).astype(np.int64)]
            inputs = values[sampled[:, None]+np.arange(32)]
            delta = np.diff(inputs, axis=1)
            variation = np.mean(abs(delta), axis=(1, 2))
            stats = dict(scope='T input windows only; deterministic evenly spaced sample up to 8192; no target for these statistics',
                         sampled_windows=len(sampled), channel_mean_abs_first_difference=np.mean(abs(delta), axis=(0, 1)).tolist(),
                         channel_last_minus_first_rms=np.sqrt(np.mean((inputs[:, -1]-inputs[:, 0])**2, axis=0)).tolist(),
                         channel_temporal_std_mean=np.mean(np.std(inputs, axis=1), axis=0).tolist(),
                         input_variation_quantiles=np.quantile(variation, [0, .25, .5, .75, .9, 1]).tolist())
        tasks.append(dict(task=task, receipt=receipt, train_input_statistics=stats,
                          B0=results[task+'_B0']['metrics'], parent_FIX=results[task+'_FIX']['metrics']))
    state = dict(tasks=tasks, known_reference_comparisons=paired,
                 warning='Two groups of one historical dataset, not independent sources. Selection-task responses are withheld. Historical source-half is the fixed parent, not a new finding.',
                 baseline_protocol_sha256=sha(root/'baseline_protocol.json'))
    write(root/'construction_state.json', state)
    report = dict(status='PASS', runs=6, asset_hashes=True, metric_recomputation=True, paired_initialization_and_draws=True,
                  source_half_vs_B0=paired, training_process_seconds=sum(x['process_seconds'] for x in results.values()),
                  process_seconds=time.monotonic()-started, new_trainings=6, cumulative_completed_trainings=243, test_access=False)
    write(root/'baseline_audit.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
