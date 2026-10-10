"""Bounded covariate-sampling study with frozen progression and controls."""
import json
import subprocess
import sys
import time
from run_research import ROOT, write_json
from covariate_sampling import FOLDER, ROUND_START, ROUND_LIMIT, used_seconds


def main():
    path = FOLDER / 'progress.json'; assert not path.exists()
    assert abs(used_seconds() - ROUND_START) < 1e-6, 'Reconcile original budget before starting'
    progress = {'phase': 'starting', 'rows': [], 'auxiliary_runs': [], 'winner': None, 'adopt_default': False,
                'round_start_seconds': ROUND_START, 'round_limit_seconds': ROUND_LIMIT}
    def save(phase):
        progress.update(phase=phase, budget_used_seconds=used_seconds()); write_json(path, progress)
    def launch(name, command):
        folder = ROOT / 'research_runs/results' / name; assert not folder.exists()
        allowance = ROUND_LIMIT - used_seconds() - 20
        if allowance < 100: raise TimeoutError('Insufficient remaining study budget')
        save('running_' + name); begun = time.monotonic()
        try:
            with (FOLDER / (name + '.log')).open('xb') as log:
                subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=allowance)
        except BaseException:
            folder.mkdir(exist_ok=True); status = folder / 'status.json'
            previous = json.loads(status.read_text()) if status.exists() else {}
            write_json(status, {**previous, 'phase': 'failed_or_budget_stopped',
                'elapsed_seconds': max(previous.get('elapsed_seconds', 0), time.monotonic() - begun)})
            raise
        return json.loads((folder / 'result.json').read_text())
    def fit(name, origin):
        result = launch(name, ['scripts/covariate_sampling.py', '--run-id', name, '--origin', str(origin)])
        progress['auxiliary_runs'].append(name); save('completed_' + name)
        print(json.dumps({'auxiliary': name, 'profiles': result['profiles']}), flush=True)
        return name
    def run(name, origin, policy, scorer, seed=2026, selected=None):
        command = ['scripts/run_covariate.py', '--run-id', name, '--origin', str(origin), '--policy', policy,
                   '--score-run', scorer, '--seed', str(seed)]
        if selected: command += ['--selected-policy', selected]
        result = launch(name, command)
        if origin < 1:
            baseline = 'T000_prefix60_full' if origin == .6 else 'T005_prefix80_full'
            metrics = result['future']['groups']['all']
        else:
            baseline = {2026: 'S000_uniform_s2026', 2027: 'S005_uniform_s2027', 2028: 'S010_uniform_s2028'}[seed]
            metrics = result['groups']['all']
        ref = json.loads((ROOT / 'research_runs/results' / baseline / 'result.json').read_text())
        b = ref['future']['groups']['all'] if origin < 1 else ref['groups']['all']
        row = {'run_id': name, 'origin': origin, 'policy': policy, 'seed': seed, 'baseline': baseline, **metrics,
            'NMSE_reduction_pct': 100 * (1 - metrics['NMSE'] / b['NMSE']),
            'NMAE_reduction_pct': 100 * (1 - metrics['NMAE'] / b['NMAE'])}
        progress['rows'].append(row); save('completed_' + name); print(json.dumps(row), flush=True)
        return row
    try:
        scorer = fit('P003_covariate_prefix60', .6)
        a = run('T045_prefix60_covariate_global', .6, 'global', scorer)
        b = run('T046_prefix60_covariate_state_matched', .6, 'state_matched', scorer)
        best = min([a, b], key=lambda r: r['NMSE']); winner = progress['winner'] = best['policy']
        control = run('T047_prefix60_covariate_shuffle', .6, 'shuffle', scorer, selected=winner)
        if best['NMSE_reduction_pct'] <= 0 or best['NMSE'] >= control['NMSE']:
            save('stopped_no_prefix60_controlled_signal'); return
        scorer = fit('P004_covariate_prefix80', .8)
        second = run('T048_prefix80_covariate_' + winner, .8, winner, scorer)
        second_control = run('T049_prefix80_covariate_shuffle', .8, 'shuffle', scorer, selected=winner)
        if second['NMSE_reduction_pct'] <= 0 or second['NMSE'] >= second_control['NMSE']:
            save('stopped_no_prefix80_controlled_signal'); return
        progress['internal_signal_confirmed'] = True
        scorer = fit('P005_covariate_full', 1.)
        formal = [run('S038_covariate_' + winner + '_s2026', 1., winner, scorer)]
        if formal[0]['NMSE_reduction_pct'] <= 0:
            save('stopped_no_formal_signal'); return
        formal.append(run('S039_covariate_' + winner + '_s2027', 1., winner, scorer, 2027))
        formal.append(run('S040_covariate_' + winner + '_s2028', 1., winner, scorer, 2028))
        refs = [json.loads((ROOT / 'research_runs/results' / r['baseline'] / 'result.json').read_text())['groups']['all'] for r in formal]
        mse = sum(r['NMSE'] for r in formal) / 3; mae = sum(r['NMAE'] for r in formal) / 3
        mse_gain = 100 * (1 - mse / (sum(r['NMSE'] for r in refs) / 3))
        mae_gain = 100 * (1 - mae / (sum(r['NMAE'] for r in refs) / 3))
        progress.update(mean_NMSE=mse, mean_NMAE=mae, mean_NMSE_reduction_pct=mse_gain, mean_NMAE_reduction_pct=mae_gain,
                        adopt_default=mse_gain >= 1 and mae_gain >= 0 and all(r['NMSE_reduction_pct'] > 0 for r in formal))
        save('completed_formal_confirmation')
    except BaseException as error:
        progress['error'] = str(error); save('failed_or_budget_stopped'); raise


if __name__ == '__main__': main()
