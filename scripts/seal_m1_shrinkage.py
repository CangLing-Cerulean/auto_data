"""Verify and package only the completed shrinkage stage; no experiment execution."""
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


def main(stage, source):
    stage, source = stage.resolve(strict=True), source.resolve(strict=True)
    assert stage.name == '20261022_m1_shrinkage_r1'
    prediction, lock, audit, evaluation = [read(stage / n) for n in ('predictions.json', 'source_lock.json', 'audit.json', 'evaluation.json')]
    assert sha(stage / 'predictions.json') == read(stage / 'prediction_lock.json')['sha256'] == audit['predictions_sha256'] == evaluation['predictions_sha256']
    assert sha(stage / 'source_lock.json') == prediction['source_lock_sha256'] == audit['source_lock_sha256']
    assert sha(stage / 'audit.json') == evaluation['audit_sha256'] and audit['status'] == 'PASS'
    assert sha(source / 'labels.json') == evaluation['labels_sha256']
    for name, digest in lock['sha256'].items():
        assert sha(stage / name) == digest
    for name, digest in prediction['source_input_sha256'].items():
        assert sha(source / name) == digest
    for name, digest in lock['evaluation_metadata_sha256'].items():
        assert sha(source / name) == digest
    paths = []
    for r in prediction['records']:
        for record in r['methods'].values():
            for name, digest in record['assets_sha256'].items():
                path = stage / record['directory'] / name
                assert sha(path) == digest
                paths.append(path)
    assert len(paths) == len(set(paths)) == 864
    assert evaluation['candidate_for_future_validation'] == 'CHANNEL_SHRINK'
    assert evaluation['gates']['SCALAR_SHRINK']['pass_all'] and evaluation['gates']['CHANNEL_SHRINK']['pass_all']
    verification = dict(status='PASS', locked_source_files=len(lock['sha256']),
        source_files=len(prediction['source_input_sha256']), forecast_files=len(paths),
        verified_utc=datetime.now(timezone.utc).isoformat(), new_full_trainings=0, historical_full_completed=593)
    with (stage / 'verification.json').open('x', encoding='utf-8') as stream:
        json.dump(verification, stream, indent=2)
    files = []
    for path in sorted(stage.rglob('*')):
        relative = path.relative_to(stage)
        assert not path.is_symlink()
        if path.is_file() and not {'archives', '__pycache__'}.intersection(relative.parts):
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
    print(json.dumps(dict(verification=verification, archive=receipt)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    main(args.stage, args.source)
