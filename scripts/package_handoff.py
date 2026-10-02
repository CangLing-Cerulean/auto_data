"""Package the frozen dataset and research assets without changing their bytes."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = 'handoff-20261002'
OUTPUT = ROOT / '.handoff_packages'
MANIFEST = ROOT / 'research_runs/handoff_assets.json'
BASE_URL = f'https://github.com/CangLing-Cerulean/auto_data/releases/download/{TAG}'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if MANIFEST.exists():
        raise FileExistsError('This handoff is already frozen; do not overwrite it.')
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_dataset.py')], check=True)
    artifacts = sorted(p for p in (ROOT / 'research_runs').rglob('*')
                       if p.is_file() and (p.suffix in {'.pt', '.npz', '.log'}
                                          or p.name.endswith('.csv.gz')))
    groups = {
        'metropt3-v1-development.zip': [ROOT / f'data/metropt3_v1/{s}.csv'
                                      for s in ('train', 'validation')],
        'metropt3-v1-test.zip': [ROOT / 'data/metropt3_v1/test.csv'],
        'research-checkpoints.zip': [p for p in artifacts if p.suffix == '.pt'],
        'research-metadata.zip': [p for p in artifacts if p.suffix != '.pt'],
    }
    OUTPUT.mkdir(exist_ok=True)
    manifest = {
        'release_tag': TAG,
        'dataset_id': 'metropt3_v1',
        'dataset_manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'),
        'test_policy': 'Integrity checks only until the final evaluation protocol is frozen.',
        'assets': [],
    }
    for name, paths in groups.items():
        target = OUTPUT / name
        entries = []
        with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in paths:
                relative = path.relative_to(ROOT).as_posix()
                checksum = digest(path)
                archive.write(path, relative)
                if digest(path) != checksum:
                    raise RuntimeError(f'Source changed during packaging: {relative}')
                entries.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': checksum})
        manifest['assets'].append({
            'name': name, 'url': f'{BASE_URL}/{name}', 'bytes': target.stat().st_size,
            'sha256': digest(target), 'contains_test': name == 'metropt3-v1-test.zip',
            'files': entries,
        })
        print(f'{name}: {len(entries)} files, {target.stat().st_size:,} bytes', flush=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Frozen manifest: {MANIFEST}')


if __name__ == '__main__':
    main()
