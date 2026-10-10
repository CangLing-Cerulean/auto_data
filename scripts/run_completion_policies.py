"""Execute the fixed two-origin data policy matrix and its source diagnostics."""
import json
import subprocess
import sys
from run_research import ROOT, digest, write_json
from reducible_sampling import used_seconds
from completion_policies import MEMBER_SEEDS

FOLDER = ROOT / 'research_runs/completion_study'


def main():
    path = FOLDER / 'policy_progress.json'; assert not path.exists()
    assert json.loads((FOLDER / 'confirmation_plan_progress.json').read_text())['phase'] == 'completed'
    progress = {'phase': 'starting', 'rows': [], 'diagnostic_runs': [], 'start_used_seconds': used_seconds(),
        'protocol_sha256': digest(FOLDER / 'PROTOCOL.md'), 'policies_sha256': digest(FOLDER / 'POLICIES.md')}
    def launch(run_id, script, arguments):
        assert not (ROOT / 'research_runs/results' / run_id).exists()
        progress.update(phase='running', current_run=run_id, used_seconds=used_seconds()); write_json(path, progress)
        command = [sys.executable, 'scripts/' + script, '--run-id', run_id, *arguments]
        with (FOLDER / (run_id + '.log')).open('xb') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads((ROOT / 'research_runs/results' / run_id / 'result.json').read_text())
    try:
        for origin, score_run, teacher, first_id in ((.6, 'A100_completion_prefix60', 'T000_prefix60_full', 100),
                (.8, 'A101_completion_prefix80', 'T005_prefix80_full', 120)):
            launch(score_run, 'fit_completion_policies.py', ['--origin', str(origin), '--teacher', teacher])
            progress['diagnostic_runs'].append(score_run); write_json(path, progress)
            policies = ['oof_window', 'oof_window_shuffle', 'oof_target', 'oof_target_shuffle', 'bilevel', 'bilevel_shuffle']
            policies += [family + '_' + str(seed) for seed in MEMBER_SEEDS for family in ('coverage_core', 'supervised_core', 'coreset_random')]
            for index, policy in enumerate(policies):
                run_id = f'C{first_id + index:03d}_prefix{round(origin * 100)}_{policy}'
                result = launch(run_id, 'run_completion_policy.py', ['--origin', str(origin), '--score-run', score_run, '--policy', policy])
                baseline = 'T000_prefix60_full' if origin == .6 else 'T005_prefix80_full'
                reference = json.loads((ROOT / 'research_runs/results' / baseline / 'result.json').read_text())['future']['groups']['all']
                metric = result['future']['groups']['all']
                row = {'run_id': run_id, 'origin': origin, 'policy': policy, 'score_run': score_run, 'seed': 2026, 'baseline': baseline,
                    **metric, 'NMSE_reduction_pct': 100 * (1 - metric['NMSE'] / reference['NMSE']),
                    'NMAE_reduction_pct': 100 * (1 - metric['NMAE'] / reference['NMAE'])}
                progress['rows'].append(row); progress.update(current_run=None, used_seconds=used_seconds()); write_json(path, progress)
                print(json.dumps(row), flush=True)
        progress.update(phase='completed_two_origin_matrix', current_run=None, used_seconds=used_seconds()); write_json(path, progress)
    except BaseException as error:
        progress.update(phase='failed', error=str(error), used_seconds=used_seconds()); write_json(path, progress); raise


if __name__ == '__main__': main()
