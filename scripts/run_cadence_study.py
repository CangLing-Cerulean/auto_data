"""Bounded temporal-cadence data-selection study with matched controls."""
import json
import subprocess
import sys
import time
from run_research import ROOT, write_json
from reducible_sampling import used_seconds
from cadence_selection import FOLDER, ROUND_START, ROUND_LIMIT


def main():
    path = FOLDER / 'progress.json'; assert not path.exists()
    assert abs(used_seconds() - ROUND_START) < 1e-6
    progress = {'phase': 'starting', 'rows': [], 'winner': None, 'adopt_default': False,
        'round_start_seconds': ROUND_START, 'round_limit_seconds': ROUND_LIMIT, 'test_accessed_this_study': False}
    def save(phase):
        progress.update(phase=phase, budget_used_seconds=used_seconds()); write_json(path, progress)
    def run(name, origin, policy, seed=2026, selected=None):
        folder = ROOT / 'research_runs/results' / name; assert not folder.exists()
        allowance = ROUND_LIMIT - used_seconds() - 20; assert allowance > 100
        save('running_' + name); begun = time.monotonic()
        command = [sys.executable, 'scripts/run_cadence.py', '--run-id', name, '--origin', str(origin),
            '--policy', policy, '--seed', str(seed)]
        if selected: command += ['--selected-policy', selected]
        try:
            with (FOLDER / (name + '.log')).open('xb') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=allowance)
        except BaseException:
            folder.mkdir(exist_ok=True); status = folder / 'status.json'
            old = json.loads(status.read_text()) if status.exists() else {}
            write_json(status, {**old, 'phase': 'failed_or_budget_stopped',
                'elapsed_seconds': max(old.get('elapsed_seconds', 0), time.monotonic() - begun)})
            raise
        result = json.loads((folder / 'result.json').read_text())
        baseline = ('T000_prefix60_full' if origin == .6 else 'T005_prefix80_full') if origin < 1 else {
            2026: 'S000_uniform_s2026', 2027: 'S005_uniform_s2027', 2028: 'S010_uniform_s2028'}[seed]
        ref = json.loads((ROOT / 'research_runs/results' / baseline / 'result.json').read_text())
        metrics = result['future']['groups']['all'] if origin < 1 else result['groups']['all']
        b = ref['future']['groups']['all'] if origin < 1 else ref['groups']['all']
        row = {'run_id': name, 'origin': origin, 'policy': policy, 'selected_policy': selected, 'seed': seed,
            'baseline': baseline, **metrics, 'NMSE_reduction_pct': 100 * (1 - metrics['NMSE'] / b['NMSE']),
            'NMAE_reduction_pct': 100 * (1 - metrics['NMAE'] / b['NMAE'])}
        progress['rows'].append(row); save('completed_' + name); print(json.dumps(row), flush=True); return row
    try:
        rows = [run('T052_prefix60_cadence_drop', .6, 'drop'), run('T053_prefix60_cadence_soft', .6, 'soft')]
        best = min(rows, key=lambda r: r['NMSE']); winner = progress['winner'] = best['policy']
        control = run('T054_prefix60_cadence_shuffle', .6, 'shuffle', selected=winner)
        if best['NMSE_reduction_pct'] <= 0 or best['NMSE'] >= control['NMSE']:
            save('stopped_no_prefix60_controlled_signal'); return
        second = run('T055_prefix80_cadence_' + winner, .8, winner)
        second_control = run('T056_prefix80_cadence_shuffle', .8, 'shuffle', selected=winner)
        if second['NMSE_reduction_pct'] <= 0 or second['NMSE'] >= second_control['NMSE']:
            save('stopped_no_prefix80_controlled_signal'); return
        progress['internal_signal_confirmed'] = True
        formal = [run('S044_cadence_' + winner + '_s2026', 1., winner)]
        if formal[0]['NMSE_reduction_pct'] <= 0:
            save('stopped_no_formal_signal'); return
        formal_control = run('S047_cadence_shuffle_s2026', 1., 'shuffle', selected=winner)
        if formal[0]['NMSE'] >= formal_control['NMSE']:
            save('stopped_no_formal_controlled_signal'); return
        formal += [run('S045_cadence_' + winner + '_s2027', 1., winner, 2027),
                   run('S046_cadence_' + winner + '_s2028', 1., winner, 2028)]
        refs = [json.loads((ROOT / 'research_runs/results' / r['baseline'] / 'result.json').read_text())['groups']['all'] for r in formal]
        mse = sum(r['NMSE'] for r in formal) / 3; mae = sum(r['NMAE'] for r in formal) / 3
        gain = 100 * (1 - mse / (sum(r['NMSE'] for r in refs) / 3))
        mae_gain = 100 * (1 - mae / (sum(r['NMAE'] for r in refs) / 3))
        progress.update(mean_NMSE=mse, mean_NMAE=mae, mean_NMSE_reduction_pct=gain, mean_NMAE_reduction_pct=mae_gain,
            adopt_default=gain >= 1 and mae_gain >= 0 and all(r['NMSE_reduction_pct'] > 0 for r in formal))
        save('completed_formal_confirmation')
    except BaseException as error:
        progress['error'] = str(error); save('failed_or_budget_stopped'); raise


if __name__ == '__main__': main()
