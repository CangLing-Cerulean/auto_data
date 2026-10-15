"""Freeze original parents, all original A/B/C windows and bounded pilot code."""
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research_runs/agent_research/intervention_diagnosis'
OUT = BASE/'20261015_adam_sampling_pilot_r1'
OLD = BASE/'20261015_fourgroup_m1_r1'
CODE = ('adam_sampling_pilot.py', 'audit_adam_sampling_pilot.py',
    'launch_adam_sampling_pilot.py', 'register_adam_sampling_pilot.py',
    'deploy_adam_sampling_pilot.py', 'deploy_fourgroup_m1.py',
    'adam_risk_sampling.py', 'fourgroup_runtime.py', 'response_proxy.py', 'run_research.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    archive, sync = read(OLD/'forecast_archive_receipt.json'), read(OLD/'forecast_sync_receipt.json')
    assert archive['status'] == 'PASS' and sync['archive_members'] == 4085
    members = {item['path']: item for item in archive['members']}
    pattern = re.compile(r'inputs/runs/DEV_G[123]_full_PARENT_s202610170[123]/(?:initial\.pt|step_(?:500|1000|1500|2000|2500|3000)\.pt)$')
    names = sorted(n for n in members if pattern.fullmatch(n)
        or n in [f'inputs/prepared/G{i}_ABC.npz' for i in (1,2,3)]+['inputs/data/manifest.json'])
    assert len(names) == 67
    entries, files = [], []
    for name in names:
        path, item = OLD/'archives/verified_forecast'/name, members[name]
        assert path.resolve() == path and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        entries.append(dict(item, source_archive_member=name)); files.append((path, name))
    lock = dict(status='verified_original_parent_whitelist', files=entries,
        source_archive=sync['assets'][0], source_archive_receipt_sha256=sha(OLD/'forecast_archive_receipt.json'),
        source_sync_receipt_sha256=sha(OLD/'forecast_sync_receipt.json'),
        numeric_arrays_enabled=['A_values','A_starts','A_cycles','B_values','B_starts','C_values','C_starts'],
        old_candidate_models_enabled=False, D_test_enabled=False)
    write(OUT/'input_lock.json', lock)
    private = OUT/'private'; private.mkdir(exist_ok=True)
    bundle = private/'inputs.tar.gz'
    with tarfile.open(bundle, 'x:gz') as packed:
        for path, name in files:
            packed.add(path, arcname=name, recursive=False)
    reserve = 192*1024*1024
    required = int((sum(x['bytes'] for x in entries)+reserve)*2*1.2)+64*1024*1024
    write(OUT/'storage_estimate.json', dict(input_bytes=sum(x['bytes'] for x in entries),
        output_metadata_and_arrays_reserve=reserve, archive_and_transfer_factor=2,
        margin_fraction=.2, probe_bytes=64*1024*1024, required_free_bytes=required,
        saved_new_or_replayed_states=252, saved_population_gradient_arrays=0))
    old_plan = read(OLD/'plan.json')
    code = {name: sha(ROOT/'scripts'/name) for name in CODE}
    for name in ('run_research.py','response_proxy.py','fourgroup_runtime.py'):
        assert code[name] == members[name]['sha256']
    for name in ('PROTOCOL.md','DECISION_SCOPE.md'):
        code[name] = sha(OUT/name)
    plan = dict(stage='adam_sampling_pilot_r1',
        output_root='/root/auto_data_experiments/20261015_adam_sampling_pilot_r1',
        seeds=old_plan['seeds'], tasks=old_plan['tasks'], training=old_plan['training'],
        methods=['adam_risk','gradient_norm','geometry'], refresh_steps=[500,1000,1500,2000,2500],
        floor_fraction=.5, maximum_new_models=27, maximum_parent_replays=9, maximum_audit_replays=9,
        preflight_parent_states=9, expected_environment=old_plan['expected_environment'],
        gates=dict(reference_rtol=1e-9, reference_atol=1e-11, gain_percent=1.,
            control_increment_pp=.10, remaining_harm_percent=.25, required_tasks=2, maximum_cost_ratio=1.10),
        code_sha256=code, input_lock_sha256=sha(OUT/'input_lock.json'),
        storage_estimate_sha256=sha(OUT/'storage_estimate.json'),
        input_bundle=dict(name=bundle.name, bytes=bundle.stat().st_size, sha256=sha(bundle)),
        preflight_min_free_bytes=required, runtime_min_free_bytes=256*1024*1024,
        cumulative_before=434, full_training_enabled=True, B_access=True, D_access=False, test_access=False,
        failure_action='end current method; assess analysis-paper evidence, no automatic new candidate')
    write(OUT/'plan.json', plan)
    print(json.dumps(dict(input_files=len(entries), bundle_bytes=bundle.stat().st_size,
        required_free_bytes=required, plan_sha256=sha(OUT/'plan.json'))))


if __name__ == '__main__':
    main()
