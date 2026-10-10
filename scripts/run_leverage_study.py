"""Leverage sampling with fixed controls, gates, and the existing shared budget."""
import json
import subprocess
import sys
import time
from run_research import ROOT, write_json
from reducible_sampling import used_seconds
from leverage_sampling import FOLDER, ROUND_START, ROUND_LIMIT, STAGE_START


def main():
    path = FOLDER / 'progress.json'; assert not path.exists()
    assert abs(used_seconds() - STAGE_START) < 1e-6
    assert json.loads((ROOT / 'research_runs/cadence_selection/progress.json').read_text())['phase'] == 'stopped_no_prefix60_controlled_signal'
    progress = {'phase': 'starting', 'rows': [], 'diagnostic_runs': [], 'winner': None, 'adopt_default': False,
        'round_start_seconds': ROUND_START, 'stage_start_seconds': STAGE_START, 'round_limit_seconds': ROUND_LIMIT,
        'test_accessed_this_study': False}
    def save(phase):
        progress.update(phase=phase, budget_used_seconds=used_seconds()); write_json(path, progress)
    def launch(name, command):
        folder = ROOT / 'research_runs/results' / name; assert not folder.exists()
        allowance = ROUND_LIMIT - used_seconds() - 20; assert allowance > 100
        save('running_' + name); begun = time.monotonic()
        try:
            with (FOLDER / (name + '.log')).open('xb') as log:
                subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=allowance)
        except BaseException:
            folder.mkdir(exist_ok=True); status = folder / 'status.json'
            old = json.loads(status.read_text()) if status.exists() else {}
            write_json(status, {**old, 'phase': 'failed_or_budget_stopped',
                'elapsed_seconds': max(old.get('elapsed_seconds', 0), time.monotonic() - begun)})
            raise
        return json.loads((folder / 'result.json').read_text())
    def fit(name, origin):
        result = launch(name, ['scripts/leverage_sampling.py', '--run-id', name, '--origin', str(origin)])
        progress['diagnostic_runs'].append(name); save('completed_' + name)
        print(json.dumps({'diagnostic': name, 'windows': result['windows'], 'score_quantiles': result['score_quantiles']}), flush=True)
        return name
    def run(name, origin, policy, scorer, seed=2026, selected=None):
        command = ['scripts/run_leverage.py', '--run-id', name, '--origin', str(origin), '--policy', policy,
            '--score-run', scorer, '--seed', str(seed)]
        if selected: command += ['--selected-policy', selected]
        result = launch(name, command)
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
        scorer = fit('A016_leverage_prefix60', .6)
        rows = [run('T057_prefix60_leverage_uncorrected', .6, 'uncorrected', scorer),
                run('T058_prefix60_leverage_corrected', .6, 'corrected', scorer)]
        best = min(rows, key=lambda r: r['NMSE']); winner = progress['winner'] = best['policy']
        control = run('T059_prefix60_leverage_shuffle', .6, 'shuffle', scorer, selected=winner)
        if best['NMSE_reduction_pct'] <= 0 or best['NMSE'] >= control['NMSE']:
            save('stopped_no_prefix60_controlled_signal'); return
        scorer = fit('A017_leverage_prefix80', .8)
        second = run('T060_prefix80_leverage_' + winner, .8, winner, scorer)
        second_control = run('T061_prefix80_leverage_shuffle', .8, 'shuffle', scorer, selected=winner)
        if second['NMSE_reduction_pct'] <= 0 or second['NMSE'] >= second_control['NMSE']:
            save('stopped_no_prefix80_controlled_signal'); return
        progress['internal_signal_confirmed'] = True
        scorer = fit('A018_leverage_full', 1.)
        formal = [run('S048_leverage_' + winner + '_s2026', 1., winner, scorer)]
        if formal[0]['NMSE_reduction_pct'] <= 0:
            save('stopped_no_formal_signal'); return
        control = run('S051_leverage_shuffle_s2026', 1., 'shuffle', scorer, selected=winner)
        if formal[0]['NMSE'] >= control['NMSE']:
            save('stopped_no_formal_controlled_signal'); return
        formal += [run('S049_leverage_' + winner + '_s2027', 1., winner, scorer, 2027),
                   run('S050_leverage_' + winner + '_s2028', 1., winner, scorer, 2028)]
        refs = [json.loads((ROOT / 'research_runs/results' / r['baseline'] / 'result.json').read_text())['groups']['all'] for r in formal]
        mse = sum(r['NMSE'] for r in formal) / 3; mae = sum(r['NMAE'] for r in formal) / 3
        gain = 100 * (1 - mse / (sum(r['NMSE'] for r in refs) / 3)); mae_gain = 100 * (1 - mae / (sum(r['NMAE'] for r in refs) / 3))
        progress.update(mean_NMSE=mse, mean_NMAE=mae, mean_NMSE_reduction_pct=gain, mean_NMAE_reduction_pct=mae_gain,
            adopt_default=gain >= 1 and mae_gain >= 0 and all(r['NMSE_reduction_pct'] > 0 for r in formal))
        save('completed_formal_confirmation')
    except BaseException as error:
        progress['error'] = str(error); save('failed_or_budget_stopped'); raise


if __name__ == '__main__': main()
