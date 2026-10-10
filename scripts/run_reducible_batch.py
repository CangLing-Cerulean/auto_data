"""Feedback-gated reference-loss study within the original session budget."""
import json
import subprocess
import sys
import time
from run_research import ROOT, write_json
from reducible_sampling import FOLDER, ROUND_LIMIT, used_seconds, REFERENCE_SEED


def main():
    path = FOLDER / 'progress.json'
    assert not path.exists(), 'Inspect existing progress before any resume'
    progress = {'phase': 'starting', 'rows': [], 'auxiliary_runs': [], 'winner': None,
                'round_limit_seconds': ROUND_LIMIT, 'adopt_default': False}
    def save(phase):
        progress.update(phase=phase, budget_used_seconds=used_seconds())
        write_json(path, progress)
    def run(name, origin, policy, reference=None, control=None, seed=2026):
        folder = ROOT / 'research_runs/results' / name
        assert not folder.exists(), 'Run ID already exists: ' + name
        allowance = ROUND_LIMIT - used_seconds() - 20
        if allowance < 180: raise TimeoutError('Insufficient remaining study time for another run')
        command = [sys.executable, 'scripts/run_reducible.py', '--run-id', name, '--origin', str(origin),
                   '--policy', policy, '--seed', str(seed)]
        if reference: command += ['--reference', reference]
        if control: command += ['--control-source', control]
        save('running_' + name); begun = time.monotonic()
        try:
            with (FOLDER / (name + '.log')).open('xb') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=allowance)
        except BaseException:
            folder.mkdir(exist_ok=True)
            status = folder / 'status.json'
            previous = json.loads(status.read_text()) if status.exists() else {}
            write_json(status, {**previous, 'phase': 'failed_or_budget_stopped',
                'elapsed_seconds': max(previous.get('elapsed_seconds', 0), time.monotonic() - begun)})
            raise
        value = json.loads((folder / 'result.json').read_text())
        if policy == 'reference':
            progress['auxiliary_runs'].append(name); save('completed_' + name)
            print(json.dumps({'auxiliary_completed': name}), flush=True); return name
        if origin < 1:
            base = 'T000_prefix60_full' if origin == .6 else 'T005_prefix80_full'
            measured = value['future']['groups']['all']
        else:
            base = {2026: 'S000_uniform_s2026', 2027: 'S005_uniform_s2027', 2028: 'S010_uniform_s2028'}[seed]
            measured = value['groups']['all']
        baseline = json.loads((ROOT / 'research_runs/results' / base / 'result.json').read_text())
        b = baseline['future']['groups']['all'] if origin < 1 else baseline['groups']['all']
        row = {'run_id': name, 'origin': origin, 'policy': policy, 'seed': seed, 'baseline': base, **measured,
            'NMSE_reduction_pct': 100 * (1 - measured['NMSE'] / b['NMSE']),
            'NMAE_reduction_pct': 100 * (1 - measured['NMAE'] / b['NMAE'])}
        progress['rows'].append(row); save('completed_' + name); print(json.dumps(row), flush=True); return row
    try:
        ref60 = run('P000_reference_prefix60', .6, 'reference', seed=REFERENCE_SEED)
        first = [run('T035_prefix60_rho', .6, 'rho', ref60),
                 run('T036_prefix60_adarho', .6, 'adarho', ref60),
                 run('T037_prefix60_hard_rank', .6, 'hard_rank')]
        winner = min(first, key=lambda r: r['NMSE'])
        progress['winner'] = winner['policy']
        control60 = run('T038_prefix60_shuffle', .6, 'shuffle', control=winner['run_id'])
        if winner['NMSE_reduction_pct'] <= 0:
            save('stopped_no_prefix60_signal'); return
        comparator = 'hard_rank' if winner['policy'] != 'hard_rank' else min(first[:2], key=lambda r: r['NMSE'])['policy']
        progress['comparator'] = comparator
        ref80 = run('P001_reference_prefix80', .8, 'reference', seed=REFERENCE_SEED)
        second = run('T039_prefix80_' + winner['policy'], .8, winner['policy'], ref80 if winner['policy'] != 'hard_rank' else None)
        other80 = run('T040_prefix80_' + comparator, .8, comparator, ref80 if comparator != 'hard_rank' else None)
        control80 = run('T041_prefix80_shuffle', .8, 'shuffle', control=second['run_id'])
        supported = (winner['NMSE'] < control60['NMSE'] and second['NMSE_reduction_pct'] > 0
                     and second['NMSE'] < control80['NMSE'] and second['NMSE'] < other80['NMSE'])
        progress['internal_signal_confirmed'] = supported
        if not supported:
            save('stopped_no_transfer_or_control_support'); return
        reffull = None
        if winner['policy'] != 'hard_rank':
            reffull = run('P002_reference_fulltrain', 1., 'reference', seed=REFERENCE_SEED)
        formal = [run('S032_' + winner['policy'] + '_s2026', 1., winner['policy'], reffull)]
        if formal[0]['NMSE_reduction_pct'] <= 0:
            save('stopped_no_formal_signal'); return
        formal.append(run('S033_' + winner['policy'] + '_s2027', 1., winner['policy'], reffull, seed=2027))
        formal.append(run('S034_' + winner['policy'] + '_s2028', 1., winner['policy'], reffull, seed=2028))
        mean_nmse = sum(r['NMSE'] for r in formal) / 3
        mean_nmae = sum(r['NMAE'] for r in formal) / 3
        baseline = [json.loads((ROOT / 'research_runs/results' / r['baseline'] / 'result.json').read_text())['groups']['all'] for r in formal]
        improvement = 100 * (1 - mean_nmse / (sum(b['NMSE'] for b in baseline) / 3))
        mae_improvement = 100 * (1 - mean_nmae / (sum(b['NMAE'] for b in baseline) / 3))
        progress.update(mean_NMSE=mean_nmse, mean_NMAE=mean_nmae, mean_NMSE_reduction_pct=improvement,
            mean_NMAE_reduction_pct=mae_improvement,
            adopt_default=improvement >= 1 and all(r['NMSE_reduction_pct'] > 0 for r in formal) and mae_improvement >= 0)
        save('completed_formal_confirmation')
    except BaseException as error:
        progress['error'] = str(error); save('failed_or_budget_stopped'); raise


if __name__ == '__main__': main()
