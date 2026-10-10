"""Run the preregistered two-model comparison and paired confirmation."""
import json
import subprocess
import sys
import time
from run_research import ROOT, write_json
from reducible_sampling import used_seconds
from run_model_exploration import FOLDER, ROUND_START, ROUND_LIMIT, BASELINES


def main():
    path = FOLDER / 'progress.json'; assert not path.exists()
    assert abs(used_seconds() - ROUND_START) < .01
    progress = {'phase': 'starting', 'rows': [], 'winner': None, 'test_eligible': False,
        'round_start_seconds': ROUND_START, 'round_limit_seconds': ROUND_LIMIT}
    def save(phase):
        progress.update(phase=phase, budget_used_seconds=used_seconds()); write_json(path, progress)
    def run(name, kind, seed):
        folder = ROOT / 'research_runs/results' / name; assert not folder.exists()
        allowance = ROUND_LIMIT - used_seconds() - 20; assert allowance > 100
        save('running_' + name); begun = time.monotonic()
        try:
            with (FOLDER / (name + '.log')).open('xb') as log:
                subprocess.run([sys.executable, 'scripts/run_model_exploration.py', '--run-id', name,
                    '--kind', kind, '--seed', str(seed)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=allowance)
        except BaseException:
            if folder.exists():
                status = folder / 'status.json'; old = json.loads(status.read_text()) if status.exists() else {}
                write_json(status, {**old, 'phase': 'failed_or_budget_stopped',
                    'elapsed_seconds': max(old.get('elapsed_seconds', 0), time.monotonic() - begun)})
            raise
        result = json.loads((folder / 'result.json').read_text())
        ref = json.loads((ROOT / 'research_runs/results' / BASELINES[seed] / 'result.json').read_text())
        metrics = result['groups']['all']; b = ref['groups']['all']
        row = {'run_id': name, 'kind': kind, 'seed': seed, 'baseline': BASELINES[seed], **metrics,
            'NMSE_reduction_pct': 100 * (1 - metrics['NMSE'] / b['NMSE']),
            'NMAE_reduction_pct': 100 * (1 - metrics['NMAE'] / b['NMAE'])}
        progress['rows'].append(row); save('completed_' + name); print(json.dumps(row), flush=True)
        return row
    try:
        candidates = [run('M000_cross_linear_s2026', 'cross_linear', 2026),
                      run('M001_residual_mlp_s2026', 'residual_mlp', 2026)]
        eligible = [r for r in candidates if r['NMSE_reduction_pct'] >= 1]
        if not eligible: save('stopped_no_validation_signal'); return
        selected = min(eligible, key=lambda r: r['NMSE'])
        winner = progress['winner'] = selected['kind']
        formal = [selected, run('M002_' + winner + '_s2027', winner, 2027), run('M003_' + winner + '_s2028', winner, 2028)]
        refs = [json.loads((ROOT / 'research_runs/results' / r['baseline'] / 'result.json').read_text())['groups']['all'] for r in formal]
        mse = sum(r['NMSE'] for r in formal) / 3; mae = sum(r['NMAE'] for r in formal) / 3
        gain = 100 * (1 - mse / (sum(r['NMSE'] for r in refs) / 3))
        progress.update(selected_runs=[r['run_id'] for r in formal], mean_NMSE=mse, mean_NMAE=mae,
            mean_NMSE_reduction_pct=gain,
            mean_NMAE_reduction_pct=100 * (1 - mae / (sum(r['NMAE'] for r in refs) / 3)),
            test_eligible=gain >= 1 and all(r['NMSE_reduction_pct'] > 0 for r in formal))
        save('completed_confirmation')
    except BaseException as error:
        progress['error'] = str(error); save('failed_or_budget_stopped'); raise


if __name__ == '__main__': main()
