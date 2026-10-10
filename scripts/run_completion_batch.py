"""Run a finite preregistered completion plan, retaining each immutable receipt."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

from run_research import ROOT, write_json, digest
from reducible_sampling import used_seconds

FOLDER = ROOT / 'research_runs/completion_study'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); args = parser.parse_args()
    assert Path(args.plan).name == args.plan
    plan_path = FOLDER / args.plan; plan = json.loads(plan_path.read_text())
    progress_path = FOLDER / (plan_path.stem + '_progress.json')
    progress = {'plan_sha256': digest(plan_path), 'phase': 'starting', 'rows': [], 'start_used_seconds': used_seconds()}
    assert not progress_path.exists()
    for item in plan:
        run_id = item['run_id']; run = ROOT / 'research_runs/results' / run_id
        assert not run.exists(), 'A planned run already exists; reconcile before resuming'
        progress.update(phase='running', current_run=run_id, used_seconds=used_seconds()); write_json(progress_path, progress)
        command = [sys.executable, 'scripts/run_completion_confirmation.py']
        for key, value in item.items(): command += ['--' + key.replace('_', '-'), str(value)]
        try:
            with (FOLDER / (run_id + '.log')).open('xb') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        except BaseException as error:
            progress.update(phase='failed', error=str(error), used_seconds=used_seconds()); write_json(progress_path, progress); raise
        result = json.loads((run / 'result.json').read_text())
        row = {**item, **result.get('future', result)['groups']['all'], 'elapsed_seconds': result['elapsed_seconds']}
        progress['rows'].append(row); progress.update(current_run=None, used_seconds=used_seconds())
        write_json(progress_path, progress); print(json.dumps(row), flush=True)
        if item['policy'] == 'uniform' and item['origin'] == 1 and item['seed'] == 2026 and item['steps'] == 3000:
            baseline = json.loads((ROOT / 'research_runs/results/S000_uniform_s2026/result.json').read_text())
            with np.load(run / 'windows.npz') as new, np.load(ROOT / 'research_runs/results/S000_uniform_s2026/sampling_weights.npz') as old:
                count_match = np.array_equal(new['draw_count'], old['draw_count'])
            difference = abs(row['NMSE'] - baseline['groups']['all']['NMSE'])
            progress['executor_verification'] = {'draw_counts_identical': count_match, 'NMSE_absolute_difference': difference}
            if not count_match or difference >= 1e-9:
                progress.update(phase='failed_executor_verification'); write_json(progress_path, progress)
                raise AssertionError('Uniform executor differs from frozen baseline')
    progress.update(phase='completed', current_run=None, used_seconds=used_seconds()); write_json(progress_path, progress)


if __name__ == '__main__': main()
