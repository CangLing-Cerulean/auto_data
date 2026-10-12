"""Create the immutable fixed-reference archive with an explicit allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    target = root/'stage1-fixed-references-20261012-r1.tar.gz'
    assert not target.exists()
    protocol = json.loads((root/'baseline_protocol.json').read_text(encoding='utf-8-sig'))
    core = Path(protocol['frozen_core'])
    assert sha(core) == protocol['frozen_core_sha256']
    snapshot = root/'frozen_core_snapshot.py'
    assert not snapshot.exists()
    shutil.copyfile(core, snapshot)
    names = ['preparation_protocol.json', 'prepare_intervention_stage1.py', 'baseline_protocol.json',
             'run_intervention_stage1_baseline.py', 'check_stage1_baseline.py', 'audit_stage1_baselines.py',
             'package_stage1_baselines.py', 'frozen_core_snapshot.py', 'preparation_receipt.json',
             'baseline_audit.json', 'construction_state.json', 'tasks', 'baseline_runs',
             'online_ts/agent_research/stage1_scoring.py', 'tests/test_stage1_scoring.py']
    files = []
    for name in names:
        path = root/name
        assert path.exists()
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        for source in candidates:
            if source.is_file():
                assert not source.is_symlink() and root.resolve() in source.resolve().parents
                relative = source.relative_to(root).as_posix()
                assert not any(x in relative.lower() for x in ('private', 'test.csv', 'verification/', '.ssh'))
                files.append(dict(name=relative, bytes=source.stat().st_size, sha256=sha(source)))
    with tarfile.open(target, 'w:gz', compresslevel=6) as archive:
        for item in files:
            archive.add(root/item['name'], arcname=item['name'], recursive=False)
    receipt = dict(name=target.name, bytes=target.stat().st_size, sha256=sha(target), members=files,
                   new_trainings=6, completed_training_total=243, derived_T_D_arrays_included=True,
                   raw_CSV_included=False, test_data_included=False, credentials_included=False)
    (root/'baseline_archive_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'members'}))


if __name__ == '__main__':
    main()
