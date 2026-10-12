"""Package auditable R2 assets; keep controller traces in a separate private archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    args = parser.parse_args()
    folder = Path(args.folder).resolve()
    project = Path(__file__).resolve().parents[1]
    destination = folder/'archives'
    destination.mkdir(exist_ok=True)
    public = {}
    private = {}
    for path in sorted((folder/'run').rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(folder).as_posix()
        parts = Path(relative).parts
        if 'inputs' in parts or '__pycache__' in parts:
            continue
        if path.suffix in ('.sqlite','.sqlite-wal','.sqlite-shm') or path.name in ('trace.jsonl','model_calls.jsonl','status.json'):
            private[relative] = path
        elif path.suffix in ('.json','.py','.pt','.npz'):
            public[relative] = path
    for name in ('PROTOCOL.md','protocol.json','AGENT_INSTRUCTIONS.md','preregistration.json',
            'memory_manifest.json','confirmation_freeze.json','confirmation_result.json','confirmation_control.json',
            'audit_summary.json','cost_summary.json','confirmation_progress.json',
            'control_semantic_review.json','actual_llm_runtime.json','launch.json',
            'confirmation_transport_recovery.json','decision_summary.json',
            'confirmation_failed_execution.json','scientific_review.json','experience_entries.json',
            'REPORT.md','RELEASE_NOTES.md'):
        if (folder/name).exists():
            public[name] = folder/name
    for path in sorted((folder/'engineering').glob('*')):
        if path.suffix in ('.json', '.py'):
            public[path.relative_to(folder).as_posix()] = path
    for path in sorted((folder/'memory').glob('*')):
        if path.is_file():
            public[path.relative_to(folder).as_posix()] = path
    for pattern in ('decision_[0-9]*.json', 'result_[0-9]*.json', 'context.json', 'engineering/*.log'):
        for path in sorted(folder.glob(pattern)):
            private[path.relative_to(folder).as_posix()] = path
    identity = json.loads((folder/'preregistration.json').read_text())
    for name, expected in identity['code'].items():
        path = Path(name)
        assert sha(path) == expected, name
        public['source_snapshot/'+path.relative_to(project).as_posix()] = path
    for name in ('collect_autonomous_memory.py','confirm_autonomous_memory.py','package_autonomous_memory.py',
            'recover_autonomous_confirmation.py'):
        path = project/'scripts'/name
        if path.exists():
            public['source_snapshot/scripts/'+name] = path
    public['source_snapshot/original_core/run_research.py'] = Path('/root/data/auto_data_research/metropt3_v1/session_20260926/scripts/run_research.py')
    public['task_metadata/task.json'] = project/'research_runs/agent_research/tasks/metropt3_dev/task.json'
    public['task_metadata/train_inputs_identity.json'] = project/'research_runs/agent_research/tasks/metropt3_dev/train_inputs_identity.json'
    receipts = []
    for label, files in [('public_assets', public), ('private_controller', private)]:
        rows = [{'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in sorted(files.items())]
        manifest_path = destination/(label+'_manifest.json')
        manifest_path.write_text(json.dumps({'kind': label, 'files': rows}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        archive = destination/('autonomous-memory-20261012-sol-high-r2-'+label+'.tar.gz')
        if archive.exists():
            raise RuntimeError('Existing archive will not be overwritten')
        with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
            tar.add(manifest_path, arcname='asset_manifest.json')
            for name, path in sorted(files.items()):
                tar.add(path, arcname=name, recursive=False)
        with tarfile.open(archive, 'r:gz') as tar:
            members = {m.name: m for m in tar.getmembers()}
            assert set(members) == {'asset_manifest.json'} | set(files)
            for row in rows:
                with tar.extractfile(members[row['path']]) as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
        receipts.append(dict(kind=label, file=archive.name, bytes=archive.stat().st_size,
            sha256=sha(archive), members=len(files)+1, publish_allowed=label=='public_assets'))
    (folder/'archive_receipt.json').write_text(json.dumps({'archives':receipts}, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'archives':receipts}), flush=True)


if __name__ == '__main__':
    main()
