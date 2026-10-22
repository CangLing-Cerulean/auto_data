"""Seal the completed transfer experiment without rerunning any learner."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main(stage):
    stage = stage.resolve(strict=True)
    assert stage.name == '20261022_transfer_diagnosis_r1'
    pred, lock, audit, evaluation = [read(stage / n) for n in ('predictions.json', 'source_lock.json', 'audit.json', 'evaluation.json')]
    assert sha(stage / 'predictions.json') == read(stage / 'prediction_lock.json')['sha256'] == audit['predictions_sha256'] == evaluation['predictions_sha256']
    assert sha(stage / 'source_lock.json') == pred['source_lock_sha256']
    assert sha(stage / 'audit.json') == evaluation['audit_sha256'] and audit['status'] == 'PASS'
    assert read(stage / 'control_recovery_local_inputs' / 'status.json')['status'] == 'completed'
    assert read(stage / 'control' / 'status.json')['status'] == 'failed'
    for name, digest in lock['sha256'].items():
        assert sha(stage / name) == digest
    models = []
    for record in pred['records']:
        for result in record['methods'].values():
            path = stage / result['model']
            assert sha(path) == result['model_sha256']
            models.append(path)
    assert len(models) == len(set(models)) == 72
    assert evaluation['h1_accuracy_support'] and not evaluation['h2_accuracy_support']
    assert not evaluation['full_development_criterion']
    verification = dict(status='PASS', verified_utc=datetime.now(timezone.utc).isoformat(),
        locked_source_files=len(lock['sha256']), models=len(models), folds=len(pred['records']),
        predictions=audit['counts']['predictions'], new_full_trainings=0, historical_full_completed=593)
    with (stage / 'verification.json').open('x', encoding='utf-8') as stream:
        json.dump(verification, stream, indent=2)
    files = []
    for path in sorted(stage.rglob('*')):
        relative = path.relative_to(stage)
        assert not path.is_symlink()
        if path.is_file() and not {'archives', 'deploy_payloads', '__pycache__'}.intersection(relative.parts):
            files.append(dict(path=relative.as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
    data = json.dumps(dict(stage=stage.name, files=files), indent=2).encode('utf-8')
    output = stage / 'archives' / f'{stage.name}_complete.tar.gz'
    output.parent.mkdir(exist_ok=True)
    with output.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz') as archive:
        info = tarfile.TarInfo('MANIFEST.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        for item in files:
            path = stage / item['path']
            assert sha(path) == item['sha256']
            archive.add(path, arcname=f'{stage.name}/{item["path"]}', recursive=False)
    with tarfile.open(output, 'r:gz') as archive:
        assert len(archive.getmembers()) == len(files) + 1
        for item in files:
            with archive.extractfile(f'{stage.name}/{item["path"]}') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
    receipt = dict(status='sealed_and_readback_verified', files=len(files), archive_bytes=output.stat().st_size,
        archive_sha256=sha(output), manifest_sha256=hashlib.sha256(data).hexdigest(),
        archive=output.relative_to(stage).as_posix(), verified_members=len(files) + 1)
    with (stage / 'archive_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(dict(verification=verification, archive=receipt)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    main(parser.parse_args().stage)
