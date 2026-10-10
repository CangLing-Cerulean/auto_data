"""Complete missing second-origin evidence without changing old recipes."""
import json
import subprocess
import sys
import time
from run_research import ROOT, write_json
from reducible_sampling import used_seconds
from covariate_sampling import ROUND_START, ROUND_LIMIT

FOLDER = ROOT / 'research_runs/support_confirmation'


def main():
    prior = json.loads((ROOT / 'research_runs/covariate_sampling/progress.json').read_text())
    assert prior['phase'] == 'stopped_no_prefix80_controlled_signal'
    path = FOLDER / 'progress.json'; assert not path.exists()
    progress = {'phase': 'starting', 'rows': [], 'winner': None, 'adopt_default': False,
        'stage_start_seconds': used_seconds(), 'round_start_seconds': ROUND_START, 'round_limit_seconds': ROUND_LIMIT,
        'reused_prefix60_runs': ['T023_prefix60_balanced', 'T042_prefix60_reshuffle']}
    def save(phase):
        progress.update(phase=phase, budget_used_seconds=used_seconds()); write_json(path, progress)
    def run(name, origin, policy, seed=2026):
        folder = ROOT / 'research_runs/results' / name; assert not folder.exists()
        allowance = ROUND_LIMIT - used_seconds() - 20
        if allowance < 100: raise TimeoutError('Insufficient remaining shared budget')
        save('running_' + name); begun = time.monotonic()
        command = [sys.executable, 'scripts/run_support_confirmation.py', '--run-id', name,
                   '--origin', str(origin), '--policy', policy, '--seed', str(seed)]
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
        baseline = 'T005_prefix80_full' if origin < 1 else {2026:'S000_uniform_s2026', 2027:'S005_uniform_s2027', 2028:'S010_uniform_s2028'}[seed]
        ref = json.loads((ROOT / 'research_runs/results' / baseline / 'result.json').read_text())
        metrics = result['future']['groups']['all'] if origin < 1 else result['groups']['all']
        b = ref['future']['groups']['all'] if origin < 1 else ref['groups']['all']
        row = {'run_id': name, 'origin': origin, 'policy': policy, 'seed': seed, 'baseline': baseline, **metrics,
            'NMSE_reduction_pct': 100 * (1 - metrics['NMSE'] / b['NMSE']),
            'NMAE_reduction_pct': 100 * (1 - metrics['NMAE'] / b['NMAE'])}
        progress['rows'].append(row); save('completed_' + name); print(json.dumps(row), flush=True); return row
    try:
        rows = [run('T050_prefix80_balanced', .8, 'balanced'), run('T051_prefix80_reshuffle', .8, 'reshuffle')]
        eligible = [r for r in rows if r['NMSE_reduction_pct'] > 0]
        if not eligible: save('stopped_no_prefix80_signal'); return
        winner = progress['winner'] = min(eligible, key=lambda r:r['NMSE'])['policy']
        formal = [run('S041_support_' + winner + '_s2026', 1., winner)]
        if formal[0]['NMSE_reduction_pct'] <= 0: save('stopped_no_formal_signal'); return
        formal.append(run('S042_support_' + winner + '_s2027', 1., winner, 2027))
        formal.append(run('S043_support_' + winner + '_s2028', 1., winner, 2028))
        refs = [json.loads((ROOT / 'research_runs/results' / r['baseline'] / 'result.json').read_text())['groups']['all'] for r in formal]
        mse = sum(r['NMSE'] for r in formal) / 3; mae = sum(r['NMAE'] for r in formal) / 3
        mse_gain = 100 * (1 - mse / (sum(r['NMSE'] for r in refs) / 3)); mae_gain = 100 * (1 - mae / (sum(r['NMAE'] for r in refs) / 3))
        progress.update(mean_NMSE=mse, mean_NMAE=mae, mean_NMSE_reduction_pct=mse_gain, mean_NMAE_reduction_pct=mae_gain,
            adopt_default=mse_gain >= 1 and mae_gain >= 0 and all(r['NMSE_reduction_pct'] > 0 for r in formal))
        save('completed_formal_confirmation')
    except BaseException as error:
        progress['error'] = str(error); save('failed_or_budget_stopped'); raise


if __name__ == '__main__': main()
