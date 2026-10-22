"""Continue the first prepared-input recovery, preserving the failed control run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(stage, payload, digest):
    assert stage.resolve() == Path('/root/data/auto_data_research/20261022_transfer_diagnosis_r1')
    failed = json.loads((stage / 'control' / 'status.json').read_text(encoding='utf-8'))
    assert failed['status'] == 'failed' and failed['phase'] == 'prepare_transfer_diagnosis.py'
    assert not any((stage / name).exists() for name in ('fold_manifest.json', 'models', 'predictions.json', 'audit.json', 'evaluation.json'))
    assert sha(payload) == digest
    control = stage / 'control_recovery_local_inputs'
    control.mkdir(exist_ok=False)
    with tarfile.open(payload, 'r:gz') as archive:
        members = archive.getmembers()
        assert members and len({m.name for m in members}) == len(members)
        for member in members:
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert path.parts[0] in {'folds', 'features.json', 'prefix_feedback.json', 'evaluation_truth.json', 'fold_manifest.json', 'local_preparation.log', 'PREPARATION_RECOVERY.md'}
            assert not (stage / path).exists()
        archive.extractall(stage, members=members, filter='data')
    manifest = json.loads((stage / 'fold_manifest.json').read_text(encoding='utf-8'))
    assert sha(stage / 'source_lock.json') == manifest['source_lock_sha256']
    assert len(manifest['folds']) == 18
    for entry in manifest['folds']:
        assert sha(stage / entry['path']) == entry['sha256']
    for name, field in (('features.json', 'features_sha256'), ('prefix_feedback.json', 'feedback_sha256'), ('evaluation_truth.json', 'truth_sha256')):
        assert sha(stage / name) == manifest[field]
    status = dict(status='running', phase=None, started_utc=datetime.now(timezone.utc).isoformat(),
                  original_failure='control/status.json', input_payload_sha256=digest,
                  preparation_environment=manifest['environment'])

    def save():
        (control / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')

    for name in ('run_transfer_diagnosis.py', 'audit_transfer_diagnosis.py', 'evaluate_transfer_diagnosis.py'):
        status['phase'] = name
        save()
        with (control / (name + '.log')).open('x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, str(stage / 'source_snapshot' / name), '--stage', str(stage)],
                                    stdout=log, stderr=subprocess.STDOUT)
        status['returncode'] = result.returncode
        if result.returncode:
            status['status'] = 'failed'
            save()
            raise SystemExit(result.returncode)
    status.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    main(args.stage, args.payload, args.sha256)
