"""Two batch-organization hypotheses, with temporal and formal progression."""
import json
import subprocess
import sys
import time
from run_research import ROOT, write_json
from reducible_sampling import ROUND_LIMIT, used_seconds

FOLDER = ROOT / 'research_runs/batch_sampling'


def main():
    prior = json.loads((ROOT / 'research_runs/reducible_sampling/progress.json').read_text())
    assert prior['phase'] == 'stopped_no_prefix60_signal'
    path = FOLDER / 'progress.json'; assert not path.exists()
    progress = {'phase': 'starting', 'rows': [], 'winner': None, 'adopt_default': False,
                'stage_start_seconds': used_seconds(), 'round_limit_seconds': ROUND_LIMIT}
    def save(phase):
        progress.update(phase=phase, budget_used_seconds=used_seconds()); write_json(path, progress)
    def run(name, origin, policy, seed=2026):
        folder = ROOT / 'research_runs/results' / name; assert not folder.exists()
        allowance = ROUND_LIMIT - used_seconds() - 20
        if allowance < 100: raise TimeoutError('Not enough remaining study time for another run')
        command = [sys.executable, 'scripts/run_batch_sampling.py', '--run-id', name, '--origin', str(origin),
                   '--policy', policy, '--seed', str(seed)]
        save('running_' + name); begun = time.monotonic()
        try:
            with (FOLDER / (name + '.log')).open('xb') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=allowance)
        except BaseException:
            folder.mkdir(exist_ok=True); status = folder / 'status.json'
            previous = json.loads(status.read_text()) if status.exists() else {}
            write_json(status, {**previous, 'phase': 'failed_or_budget_stopped',
                'elapsed_seconds': max(previous.get('elapsed_seconds', 0), time.monotonic() - begun)})
            raise
        result = json.loads((folder / 'result.json').read_text())
        if origin < 1:
            base = 'T000_prefix60_full' if origin == .6 else 'T005_prefix80_full'
            metrics = result['future']['groups']['all']
        else:
            base = {2026: 'S000_uniform_s2026', 2027: 'S005_uniform_s2027', 2028: 'S010_uniform_s2028'}[seed]
            metrics = result['groups']['all']
        baseline = json.loads((ROOT / 'research_runs/results' / base / 'result.json').read_text())
        b = baseline['future']['groups']['all'] if origin < 1 else baseline['groups']['all']
        row = {'run_id': name, 'origin': origin, 'policy': policy, 'seed': seed, 'baseline': base, **metrics,
            'NMSE_reduction_pct': 100 * (1 - metrics['NMSE'] / b['NMSE']),
            'NMAE_reduction_pct': 100 * (1 - metrics['NMAE'] / b['NMAE']),
            'unique_windows_presented': result['unique_windows_presented'], 'unique_fraction': result['unique_fraction']}
        progress['rows'].append(row); save('completed_' + name); print(json.dumps(row), flush=True); return row
    try:
        a = run('T042_prefix60_reshuffle', .6, 'reshuffle')
        b = run('T043_prefix60_time_stratified', .6, 'time_stratified')
        winner = min([a, b], key=lambda r: r['NMSE']); progress['winner'] = winner['policy']
        if winner['NMSE_reduction_pct'] <= 0: save('stopped_no_prefix60_signal'); return
        second = run('T044_prefix80_' + winner['policy'], .8, winner['policy'])
        if second['NMSE_reduction_pct'] <= 0: save('stopped_no_prefix80_signal'); return
        progress['internal_signal_confirmed'] = True
        formal = [run('S035_' + winner['policy'] + '_s2026', 1., winner['policy'])]
        if formal[0]['NMSE_reduction_pct'] <= 0: save('stopped_no_formal_signal'); return
        formal.append(run('S036_' + winner['policy'] + '_s2027', 1., winner['policy'], 2027))
        formal.append(run('S037_' + winner['policy'] + '_s2028', 1., winner['policy'], 2028))
        baseline = [json.loads((ROOT / 'research_runs/results' / r['baseline'] / 'result.json').read_text())['groups']['all'] for r in formal]
        mean_nmse = sum(r['NMSE'] for r in formal) / 3; mean_nmae = sum(r['NMAE'] for r in formal) / 3
        mse_gain = 100 * (1 - mean_nmse / (sum(x['NMSE'] for x in baseline) / 3))
        mae_gain = 100 * (1 - mean_nmae / (sum(x['NMAE'] for x in baseline) / 3))
        progress.update(mean_NMSE=mean_nmse, mean_NMAE=mean_nmae, mean_NMSE_reduction_pct=mse_gain,
            mean_NMAE_reduction_pct=mae_gain, adopt_default=mse_gain >= 1 and mae_gain >= 0 and all(r['NMSE_reduction_pct'] > 0 for r in formal))
        save('completed_formal_confirmation')
    except BaseException as error:
        progress['error'] = str(error); save('failed_or_budget_stopped'); raise


if __name__ == '__main__': main()
