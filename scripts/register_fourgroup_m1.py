"""Build a create-only stage from verified archive members; never read old D.

The transport bundle contains only whitelisted parent/ABC assets and base train.
Its full member identity is locked before any server execution.
"""
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research_runs/agent_research/intervention_diagnosis'
OUT = BASE / '20261015_fourgroup_m1_r1'
DESIGN = BASE / '20261015_response_methods_protocol_r1'
REMOTE = '/root/auto_data_experiments/20261015_fourgroup_m1_r1'
CODE = ('fourgroup_runtime.py', 'fourgroup_policy.py', 'check_fourgroup_policy.py',
        'run_fourgroup_m1.py', 'audit_fourgroup_m1.py', 'launch_fourgroup_m1.py',
        'register_fourgroup_m1.py', 'deploy_fourgroup_m1.py',
        'response_proxy.py', 'run_research.py', 'source_dlinear_adapter.py', 'source_quadratic.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    OUT.mkdir(exist_ok=True)
    config = read(DESIGN / 'protocol.json')
    previous = read(BASE / '20261013_proxy_selection_r1/plan.json')
    archive_root = BASE / '20261013_response_proxy_development_r1'
    archive_receipt = read(archive_root / 'failure_archive_receipt.json')
    verified = archive_root / 'archives/verified'
    files, bundle_sources = [], []
    allowed = re.compile(r'runs/DEV_G[123]_(full|proxy)_PARENT_s202610170[123]/[^/]+$')
    for name, digest in previous['inputs_sha256'].items():
        if name not in [f'prepared/G{i}_ABC.npz' for i in (1, 2, 3)] and not allowed.fullmatch(name):
            continue
        path = verified / name
        assert path.resolve() == path and path.is_file() and sha(path) == digest, name
        assert archive_receipt['members_sha256'][name] == digest
        relative = 'inputs/' + name
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest,
            archive_member=name, source='response_proxy_development_failed_audit_20261013.tar.gz'))
        bundle_sources.append((path, relative))
    assert len(files) == 219
    prior_preparation = verified / 'preparation.json'
    assert sha(prior_preparation) == previous['inputs_sha256']['preparation.json']
    assert sha(prior_preparation) == archive_receipt['members_sha256']['preparation.json']
    files.append(dict(path='inputs/prior_preparation.json', bytes=prior_preparation.stat().st_size,
        sha256=sha(prior_preparation), archive_member='preparation.json',
        source='response_proxy_development_failed_audit_20261013.tar.gz'))
    bundle_sources.append((prior_preparation, 'inputs/prior_preparation.json'))
    for name in ('manifest.json', 'train.csv'):
        path = ROOT / 'data/uci_hydraulic_v1' / name
        relative = 'inputs/data/' + name
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha(path),
            source='data/uci_hydraulic_v1/' + name))
        bundle_sources.append((path, relative))
    manifest = read(ROOT / 'data/uci_hydraulic_v1/manifest.json')
    train = next(row for row in manifest['splits'] if row['name'] == 'train')
    assert next(row for row in files if row['path'].endswith('/train.csv'))['sha256'] == train['sha256']
    lock = dict(status='whitelisted_inputs_hashed', files=files,
        archive_sha256=archive_receipt['sha256'],
        archive_receipt_sha256=sha(archive_root / 'failure_archive_receipt.json'),
        original_parent_manifest_sha256=sha(BASE / '20261013_proxy_selection_r1/plan.json'),
        no_candidate_models_in_bundle=True, no_D_or_test_members=True)
    write(OUT / 'input_lock.json', lock)
    private = OUT / 'private'
    private.mkdir(exist_ok=True)
    bundle = private / 'inputs.tar.gz'
    with tarfile.open(bundle, 'x:gz') as packed:
        for path, relative in bundle_sources:
            packed.add(path, arcname=relative, recursive=False)
    # Conservative uncompressed storage model, including two complete archive-size copies.
    max_checkpoint = max(row['bytes'] for row in files if row['path'].endswith('.pt'))
    input_bytes = sum(row['bytes'] for row in files)
    # 18 replays + 72 proxy + 72 short + 72 full, initial + six K + two witnesses.
    checkpoints = (18 + 72 + 72 + 72) * 9 * max_checkpoint
    # Independent moment assets and SOURCE 6x float64 delta + float32 coordinates.
    moments = (9 + 72 + 2 * 72) * (8 * 66 * 66 + 8 * 66 * 8 + 8 * 8) * 8
    source_states = 72 * 6 * 8 * 66 * 8 * (8 + 4)
    metadata = 96 * 1024 * 1024
    outputs = input_bytes + checkpoints + moments + source_states + metadata
    # Raw residuals are reconstructible from complete states and byte hashes; no dense residual copies.
    required = int((outputs * 2 + bundle.stat().st_size) * 1.20)
    storage = dict(input_bytes=input_bytes, max_parent_checkpoint_bytes=max_checkpoint,
        checkpoint_upper_bytes=checkpoints, moments_upper_bytes=moments,
        source_coordinate_upper_bytes=source_states, metadata_reserve_bytes=metadata,
        complete_uncompressed_upper_bytes=outputs, archive_upper_bytes=outputs,
        input_transport_bytes=bundle.stat().st_size, margin_fraction=.20,
        required_free_bytes=required, residuals='reconstruct_from_saved_states_with_byte_hashes')
    write(OUT / 'storage_estimate.json', storage)
    code = {name: sha(ROOT / 'scripts' / name) for name in CODE}
    code['PROTOCOL.md'] = sha(OUT / 'PROTOCOL.md')
    assert code['response_proxy.py'] == previous['code_sha256']['response_proxy.py']
    assert code['run_research.py'] == '4f694b900ca40d2a0620da1dbe916b6c78e450dcc0c903af4a7c696fcaec3052'
    plan = dict(stage='fourgroup_m1_r1', output_root=REMOTE,
        design_commit='899a942730b3c5ad51738596c3cff64846700012',
        design_protocol_sha256=sha(DESIGN / 'protocol.json'),
        design_document_sha256=sha(DESIGN / 'PROTOCOL.md'),
        input_lock_sha256=sha(OUT / 'input_lock.json'),
        input_bundle=dict(name=bundle.name, bytes=bundle.stat().st_size, sha256=sha(bundle)),
        code_sha256=code, seeds=config['training']['seeds'], grid=config['training']['checkpoints'],
        actions=config['m1_experiment']['menu_order'],
        tasks=[dict(id=t['id'], partitions={s: t[s] for s in ('A', 'B', 'C')}) for t in config['data']['tasks']],
        training=dict(input_length=32, prediction_length=8, moving_average_kernel=25,
            batch_size=256, steps=3000, learning_rate=.001, weight_decay=0.),
        cumulative_before=418, maximum_full_children=72,
        historical_preparation_seconds=read(prior_preparation)['seconds'],
        historical_preparation_scope='old_ABC_and_D_total_conservatively_allocated_not_reexecuted',
        maximum_proxy_children=72, maximum_short_children=72,
        N1_N2_enabled=False, optional_engineering_in_separate_stage=True,
        D_access=False, test_access=False, scientific_gates=config['m1_experiment'],
        storage_estimate_sha256=sha(OUT / 'storage_estimate.json'),
        expected_environment=dict(python='3.12.9', numpy='2.4.4', torch='2.14.0+cu130',
            cuda='13.0', gpu='NVIDIA GeForce RTX 4090', platform='Linux'),
        preflight_min_free_bytes=required, runtime_min_free_bytes=256 * 1024 * 1024)
    write(OUT / 'plan.json', plan)
    print(json.dumps(dict(input_files=len(files), input_bytes=input_bytes,
        bundle_bytes=bundle.stat().st_size, required_free_bytes=required, plan_sha256=sha(OUT / 'plan.json'))))


if __name__ == '__main__':
    main()
