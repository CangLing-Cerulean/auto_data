"""Finite dependent studies; selection is frozen before any follow-up result."""
import json
import subprocess
import sys
import time
from run_research import ROOT, digest, write_json
from reducible_sampling import used_seconds
from choose_completion_families import select_families, deployed_policy, control_policy

FOLDER = ROOT / 'research_runs/completion_study'


def main():
    path = FOLDER / 'followup_progress.json'; assert not path.exists()
    progress = {'phase': 'waiting_for_long_plan', 'rows': [], 'diagnostic_runs': [], 'start_used_seconds': used_seconds(),
        'followup_protocol_sha256': digest(FOLDER / 'FOLLOWUP_PROTOCOL.md'),
        'model_transfer_protocol_sha256': digest(FOLDER / 'MODEL_TRANSFER_PROTOCOL.md')}
    write_json(path, progress)
    while True:
        long = FOLDER / 'long_plan_progress.json'
        if long.exists():
            preceding = json.loads(long.read_text())
            if preceding['phase'] == 'completed': break
            if preceding['phase'].startswith('failed'):
                progress.update(phase='stopped_dependency_failure'); write_json(path, progress)
                raise RuntimeError('Long-training plan did not complete')
        time.sleep(10)
    matrix = json.loads((FOLDER / 'policy_progress.json').read_text())
    assert matrix['phase'] == 'completed_two_origin_matrix'
    choice = select_families(matrix['rows'])
    choice['matrix_sha256'] = digest(FOLDER / 'policy_progress.json')
    choice['decision_code_sha256'] = digest(ROOT / 'scripts/choose_completion_families.py')
    decision = FOLDER / 'selection_decision.json'; assert not decision.exists(); write_json(decision, choice)
    progress['selection_decision_sha256'] = digest(decision)

    def launch(run_id, script, arguments, diagnostic=False):
        assert not (ROOT / 'research_runs/results' / run_id).exists()
        progress.update(phase='running', current_run=run_id, used_seconds=used_seconds()); write_json(path, progress)
        with (FOLDER / (run_id + '.log')).open('xb') as log:
            subprocess.run([sys.executable, 'scripts/' + script, '--run-id', run_id, *arguments], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((ROOT / 'research_runs/results' / run_id / 'result.json').read_text())
        if diagnostic: progress['diagnostic_runs'].append(run_id)
        else:
            progress['rows'].append({'run_id': run_id, 'identity': result['identity'],
                **result.get('future', result)['groups']['all'], 'elapsed_seconds': result['elapsed_seconds']})
        progress.update(current_run=None, used_seconds=used_seconds()); write_json(path, progress)
        return result

    def train(run_id, policy, source, origin=1., seed=2026, model='dlinear'):
        return launch(run_id, 'run_completion_policy.py', ['--origin', str(origin), '--score-run', source,
            '--policy', policy, '--seed', str(seed), '--model', model])

    try:
        family = choice['additional_origin_and_model_transfer_family']
        policy, control = deployed_policy(family), control_policy(family)
        launch('C090_prefix70_uniform', 'run_completion_confirmation.py', ['--origin', '.7', '--policy', 'uniform'])
        launch('A102_completion_prefix70', 'fit_completion_policies.py', ['--origin', '.7', '--teacher', 'C090_prefix70_uniform'], True)
        train('C300_prefix70_candidate', policy, 'A102_completion_prefix70', .7)
        train('C301_prefix70_control', control, 'A102_completion_prefix70', .7)
        launch('A103_completion_full', 'fit_completion_policies.py', ['--origin', '1', '--teacher', 'C000_executor_uniform_s2026'], True)
        next_id = 310; controls_done = {}
        for name in choice['formal_candidates']:
            candidate, comparison = deployed_policy(name), control_policy(name)
            train(f'C{next_id:03d}_{candidate}_s2026', candidate, 'A103_completion_full'); next_id += 1
            if comparison not in controls_done:
                controls_done[comparison] = f'C{next_id:03d}_{comparison}_s2026'
                train(controls_done[comparison], comparison, 'A103_completion_full'); next_id += 1
            for seed in (2027, 2028):
                train(f'C{next_id:03d}_{candidate}_s{seed}', candidate, 'A103_completion_full', seed=seed); next_id += 1
        launch('A104_completion_uniform_export', 'make_completion_uniform.py', ['--source', 'A103_completion_full'], True)
        assert digest(ROOT / 'scripts/forecast_models.py') == digest(ROOT / 'research_runs/results/M001_residual_mlp_s2026/model_snapshot.py')
        next_id = 400
        for seed in (2026, 2027, 2028):
            train(f'C{next_id:03d}_mlp_uniform_s{seed}', 'uniform', 'A104_completion_uniform_export', seed=seed, model='residual_mlp'); next_id += 1
            train(f'C{next_id:03d}_mlp_candidate_s{seed}', policy, 'A103_completion_full', seed=seed, model='residual_mlp'); next_id += 1
            if seed == 2026:
                train(f'C{next_id:03d}_mlp_control_s{seed}', control, 'A103_completion_full', seed=seed, model='residual_mlp'); next_id += 1
        progress.update(phase='completed', current_run=None, used_seconds=used_seconds()); write_json(path, progress)
    except BaseException as error:
        progress.update(phase='failed', error=str(error), used_seconds=used_seconds()); write_json(path, progress); raise


if __name__ == '__main__': main()
