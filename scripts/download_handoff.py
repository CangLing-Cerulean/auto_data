"""Restore SHA256-verified Release assets; never overwrite an existing file."""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def destination(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe asset path: {relative}')
    target = ROOT.joinpath(*path.parts)
    if not target.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f'Asset escapes project: {relative}')
    return target


def verified(path, expected):
    return (path.is_file() and path.stat().st_size == expected['bytes']
            and digest(path) == expected['sha256'])


def restore(asset, archive_dir, verify_only):
    missing = []
    for entry in asset['files']:
        target = destination(entry['path'])
        if target.exists():
            if not verified(target, entry):
                raise ValueError(f'Existing file differs; refusing overwrite: {target}')
        else:
            missing.append(entry)
    if not missing:
        print(f"PASS: {asset['name']} ({len(asset['files'])} files)", flush=True)
        return
    if verify_only:
        raise FileNotFoundError(f"Missing {len(missing)} files from {asset['name']}")

    cache = archive_dir / asset['name']
    archive_dir.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not verified(cache, asset):
        raise ValueError(f'Archive checksum mismatch: {cache}')
    if not cache.exists():
        print(f"Downloading {asset['name']} ({asset['bytes']:,} bytes)...", flush=True)
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'MetroPT3-handoff'})
        with tempfile.NamedTemporaryFile(dir=archive_dir, suffix='.download', delete=False) as output:
            temporary = Path(output.name)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    shutil.copyfileobj(response, output)
            except BaseException:
                output.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            if not verified(temporary, asset):
                raise ValueError(f"Downloaded archive checksum mismatch: {asset['name']}")
            temporary.rename(cache)
        finally:
            temporary.unlink(missing_ok=True)

    with zipfile.ZipFile(cache) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {e['path'] for e in asset['files']}:
            raise ValueError(f'Unexpected archive members: {cache}')
        for entry in missing:
            target = destination(entry['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.restore', delete=False) as output:
                temporary = Path(output.name)
                try:
                    with archive.open(entry['path']) as source:
                        shutil.copyfileobj(source, output)
                except BaseException:
                    output.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                if not verified(temporary, entry):
                    raise ValueError(f"Extracted file checksum mismatch: {entry['path']}")
                # Exclusive creation prevents replacing existing data, including concurrent writes.
                with target.open('xb') as output, temporary.open('rb') as source:
                    shutil.copyfileobj(source, output)
            finally:
                temporary.unlink(missing_ok=True)
    print(f"Restored {asset['name']}: {len(missing)} files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-test', action='store_true',
                        help='Restore the separate test archive for integrity checks/frozen final evaluation.')
    parser.add_argument('--verify-only', action='store_true', help='Verify existing files without downloads.')
    parser.add_argument('--archive-dir', type=Path, default=ROOT / '.handoff_packages')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'research_runs/handoff_assets.json').read_text(encoding='utf-8'))
    if digest(ROOT / 'data/metropt3_v1/manifest.json') != manifest['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest identity mismatch')
    for asset in manifest['assets']:
        if asset['contains_test'] and not args.include_test:
            continue
        restore(asset, args.archive_dir.resolve(), args.verify_only)
    if not args.include_test:
        print('Test archive skipped. Full dataset integrity verification requires --include-test.')


if __name__ == '__main__':
    main()
