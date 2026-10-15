"""Whitelist verified parent assets and freeze N1 code, cost and storage bounds."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
OUT=BASE/'20261015_n1_path_engineering_r1'
OLD=BASE/'20261015_fourgroup_m1_r1'
REMOTE='/root/auto_data_experiments/20261015_n1_path_engineering_r1/experiment'
CODE=('n1_path_runtime.py','audit_n1_paths.py','run_n1_paths.py','launch_n1_paths.py',
      'register_n1_paths.py','deploy_n1_paths.py','deploy_fourgroup_m1.py',
      'fourgroup_runtime.py','audit_fourgroup_m1.py','response_tangent.py',
      'response_tangent_torch.py','check_response_tangent_torch.py','response_proxy.py','run_research.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    archive=read(OLD/'forecast_archive_receipt.json')
    synced=read(OLD/'forecast_sync_receipt.json')
    assert archive['status']=='PASS' and synced['archive_members']==4085
    members={row['path']:row for row in archive['members']}
    assert len(members)==4085
    verified=OLD/'archives/verified_forecast'
    allowed=re.compile(r'inputs/runs/DEV_G[123]_full_PARENT_s202610170[123]/(?:initial\.pt|step_(?:500|1000|1500|2000|2500|3000)\.pt|result\.json|status\.json|timing\.json|sampling\.npz)$')
    files,sources=[],[]
    names=[n for n in members if allowed.fullmatch(n) or n in
        [f'inputs/prepared/G{i}_ABC.npz' for i in (1,2,3)]+['inputs/data/train.csv','inputs/data/manifest.json']]
    assert len(names)==104
    for name in sorted(names):
        item=members[name]
        path=verified/name
        assert path.resolve()==path and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],name
        files.append(dict(item,source_archive_member=name))
        sources.append((path,name))
    costs=OLD/'forecast_cost_table.csv'
    # Reporting tables were committed after the raw archive; bind the exact published bytes.
    commit=synced['source_commit']
    assert re.fullmatch(r'[0-9a-f]{40}',commit)
    committed=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',
        'show',commit+':'+costs.relative_to(ROOT).as_posix()],cwd=ROOT,check=True,capture_output=True).stdout
    assert hashlib.sha256(committed).hexdigest()==sha(costs)
    name='inputs/forecast_cost_table.csv'
    files.append(dict(path=name,bytes=costs.stat().st_size,sha256=sha(costs),
        source_commit=commit,source_repository_path=costs.relative_to(ROOT).as_posix()))
    sources.append((costs,name))
    lock=dict(status='verified_parent_only_whitelist',files=files,
        source_archive=synced['assets'][0],source_archive_receipt_sha256=sha(OLD/'forecast_archive_receipt.json'),
        source_sync_receipt_sha256=sha(OLD/'forecast_sync_receipt.json'),
        no_proxy_or_candidate_models=True,no_D_or_test_members=True)
    write(OUT/'input_lock.json',lock)
    bundle=OUT/'private/inputs.tar.gz'
    with tarfile.open(bundle,'x:gz') as packed:
        for path,name in sources:
            packed.add(path,arcname=name,recursive=False)
    max_pt=max(r['bytes'] for r in files if r['path'].endswith('.pt'))
    inputs=sum(r['bytes'] for r in files)
    # Nine 11-state paths; nine 3-state preflight witnesses; sixteen 7-state FD trajectories.
    checkpoints=(9*11+9*3+16*7)*max_pt
    tangents=(9*11+9*3)*(3*8*66*8*4*8+8*66*8+4096)
    metadata=96*1024*1024
    outputs=inputs+checkpoints+tangents+metadata
    required=int((outputs*2+bundle.stat().st_size)*1.20)+64*1024*1024
    storage=dict(input_bytes=inputs,max_parent_checkpoint_bytes=max_pt,
        checkpoint_upper_bytes=checkpoints,tangent_upper_bytes=tangents,metadata_reserve_bytes=metadata,
        complete_uncompressed_upper_bytes=outputs,archive_upper_bytes=outputs,
        input_transport_bytes=bundle.stat().st_size,margin_fraction=.20,probe_bytes=64*1024*1024,
        required_free_bytes=required,audit_residuals='reconstructed_in_memory_no_dense_J_saved')
    write(OUT/'storage_estimate.json',storage)
    budgets={t:0. for t in ('G1','G2','G3')}
    with costs.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==9 and {(r['task'],int(r['seed'])) for r in rows}=={(t,s) for t in budgets for s in (2026101701,2026101702,2026101703)}
    for row in rows:
        budgets[row['task']]+=float(row['M1_warm_seconds'])
    old=read(OLD/'plan.json')
    code={name:sha(ROOT/'scripts'/name) for name in CODE}
    for name in ('PROTOCOL.md','RELATED_WORK.md','gpu_kernel_code_lock.json','gpu_kernel_checks.json'):
        code[name]=sha(OUT/name)
    for name in ('run_research.py','response_proxy.py','fourgroup_runtime.py','audit_fourgroup_m1.py'):
        assert code[name]==members[name]['sha256']
    kernel=read(OUT/'gpu_kernel_code_lock.json')
    assert read(OUT/'gpu_kernel_checks.json')['status']=='PASS'
    # Exact synthetic check source identities are retained without rerunning successful checks.
    assert kernel['files']==read(OUT/'gpu_kernel_checks.json')['files']
    for item in kernel['files']:
        assert code[item['name']]==item['sha256']
    plan=dict(stage='n1_path_engineering_r1',output_root=REMOTE,
        seeds=old['seeds'],grid=old['grid'],tasks=old['tasks'],training=old['training'],
        cumulative_before=418,scientific_children_enabled=False,FD_enabled=False,
        registered_maximum_FD_full_trainings=16,M1_task_budget_seconds=budgets,
        expected_environment=old['expected_environment'],code_sha256=code,
        input_lock_sha256=sha(OUT/'input_lock.json'),storage_estimate_sha256=sha(OUT/'storage_estimate.json'),
        input_bundle=dict(name=bundle.name,bytes=bundle.stat().st_size,sha256=sha(bundle)),
        preflight_min_free_bytes=required,runtime_min_free_bytes=256*1024*1024,
        D_access=False,test_access=False,preflight_required=True,independent_path_audit_required=True)
    write(OUT/'plan.json',plan)
    print(json.dumps(dict(input_files=len(files),required_free_bytes=required,bundle_bytes=bundle.stat().st_size,
        plan_sha256=sha(OUT/'plan.json'),M1_task_budget_seconds=budgets)))


if __name__=='__main__':
    main()
