"""Freeze local FTR1 code, roles and job IDs using metadata only."""
import ast
import hashlib
import json
from pathlib import Path

from register_file_transfer import registration

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261014_file_transfer_r1'
SERVER = '/root/auto_data_experiments/20261014_file_transfer_r1'
SCRIPTS = ('run_research.py', 'response_proxy.py', 'response_proxy_selection.py', 'response_proxy_existing.py',
    'audit_response_proxy_existing_r2.py', 'source_quadratic.py', 'source_dlinear_adapter.py',
    'register_file_transfer.py', 'response_proxy_file_transfer.py', 'file_transfer_source.py',
    'audit_file_transfer_core.py', 'audit_response_proxy_file_transfer.py', 'launch_file_transfer.py',
    'build_file_transfer_plan.py')
FROZEN = dict(run_research='4f694b900ca40d2a0620da1dbe916b6c78e450dcc0c903af4a7c696fcaec3052',
    response_proxy='8d06ff6c71200497205536d0204b6b865b7875e05b2b93d06f6322f06b0b68bf',
    response_proxy_selection='0cc22042dfe14c006eb4918838b3ecd76edc0b8c0043ce7d0f9161fc9d723c97',
    response_proxy_existing='89c467c5949dde34e883d5fe7d724acbe8f5a109b5b5f053a4d7f32439c7a321',
    audit_response_proxy_existing_r2='7d00acab4bd9401dbd57bcfbb13f72f118ce484eb22c4be9c9348ee523d68972',
    source_quadratic='39cd348ba93408ab510e8c506b0cc85f2f67a16da4b1c07bccc2e7a977516992',
    source_dlinear_adapter='ac130c41ba4644120a1965f483272a18afd2d92cb7d5f4abe62f0369a72a315e')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    manifest = ROOT / 'data/skab_v1/manifest.json'
    assert sha(manifest) == '372d99fd9188151d3429171c1978c3ea13a34b735bf3ec7d031d8b6845c5ecc2'
    registered = read(STAGE / 'task_registration.json')
    assert registered == registration(read(manifest))
    code = {}
    for name in SCRIPTS:
        path = ROOT / 'scripts' / name
        tree = ast.parse(path.read_text(encoding='utf-8'))
        code[name] = sha(path)
        for node in ast.walk(tree):
            imports = [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else ([node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else [])
            for imported in imports:
                if (ROOT / 'scripts' / f'{imported}.py').exists():
                    assert f'{imported}.py' in SCRIPTS, imported
    for name, digest in FROZEN.items():
        assert code[name + '.py'] == digest
    for name in ('PROTOCOL.md', 'task_registration.json', 'ENGINEERING_REVIEW.md'):
        code[name] = sha(STAGE / name)
    jobs = []
    for phase in ('parents', 'forecast', 'labels'):
        for task in registered['tasks']:
            for seed in registered['seeds']:
                pairs = [('full', 'PARENT'), ('proxy', 'PARENT')] if phase == 'parents' else (
                    [(e, a['id']) for a in task['actions'] for e in ('proxy', 'short')] if phase == 'forecast' else [('full', a['id']) for a in task['actions']])
                for engine, action in pairs:
                    jobs.append(dict(id=f'FTR1_{task["id"]}_{engine}_{action}_s{seed}', task=task['id'], seed=seed,
                                     engine=engine, action=action, phase=phase))
    assert len(jobs) == len({j['id'] for j in jobs}) == 225
    return dict(stage='file_transfer_r1', output_root=SERVER, cumulative_before=409, expected_cumulative_after=487,
        tasks=registered['tasks'], seeds=registered['seeds'], grid=[500, 1000, 1500, 2000, 2500, 3000],
        training=dict(input_length=32, prediction_length=8, moving_average_kernel=25),
        dataset=dict(id='skab_v1', sensors=registered['sensors'],
            train='/root/auto_data_experiments/20261014_existing_sources_r1/input/skab_train.csv',
            train_sha256=registered['train_sha256'],
            manifest='/root/auto_data_experiments/20261014_existing_sources_r1/input/skab_v1_manifest.json',
            manifest_sha256=registered['manifest_sha256']),
        core=SERVER + '/run_research.py', core_sha256=FROZEN['run_research'], code_sha256=code, jobs=jobs,
        source_numerical_probe=dict(blocks='first action, channel0/output0, steps500/3000, each task/seed',
                                    method='torch.matrix_exp CPU float64 augmented symmetric generator', rtol=1e-7, atol=1e-7),
        preflight_diagnostic=dict(steps=25, actions='PARENT, first, last', engines=['full', 'proxy'],
            seeds='first registered seed', tasks='all three', counted_as_full_training=False),
        activation='requires exact source commit receipt, preflight audit, and phase-specific audit/sync gates')


if __name__ == '__main__':
    plan = build()
    with (STAGE / 'plan.json').open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(stage=plan['stage'], jobs=len(plan['jobs']), code_files=len(plan['code_sha256']))))
