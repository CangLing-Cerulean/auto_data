"""Verify completed experiment receipts and archive the full evidence, without training."""
import argparse
import json
from pathlib import Path
import tarfile

from run_direct_processing import STUDY, PROJECT, load, sha, write


def main(name, metadata_only, archive_directory=None):
    destination = (Path(archive_directory) if archive_directory else STUDY.parent) / f'{name}.tar'
    if not destination.parent.is_dir():
        raise ValueError('Archive directory must already exist')
    if destination.exists():
        raise ValueError('Archive already exists; do not overwrite')
    runs = sorted((STUDY / 'runs').glob('*/result.json'))
    if not metadata_only and (len(runs) != 25 or not load(STUDY / 'confirmation_result.json')['effective_result']):
        raise ValueError('Full completion evidence not ready')
    validation = []
    for result_path in runs:
        result = load(result_path)
        folder = result_path.parent
        for asset, expected in result['assets'].items():
            if sha(folder / asset) != expected:
                raise ValueError(f'Completed asset changed: {folder.name}/{asset}')
        if load(folder / 'status.json')['phase'] != 'completed':
            raise ValueError('Completed receipt has wrong status')
        if result['parameters'] != 7920 or result['steps'] != 3000 or result['samples_presented'] != 768000:
            raise ValueError('Unequal model/updates/exposure')
        for path, expected in result['identity']['protected_assets'].items():
            if sha(path) != expected:
                raise ValueError(f'Protected source changed: {path}')
        validation.append({'run_id': result['identity']['run_id'], 'policy': result['identity']['policy'],
            'origin': result['identity']['origin'], 'seed': result['identity']['seed'],
            'NMSE': result['metrics']['groups']['all']['NMSE'], 'NMAE': result['metrics']['groups']['all']['NMAE'],
            'draw_count_sha256': result['draw_count_sha256'], 'elapsed_seconds': result['elapsed_seconds'],
            'result_sha256': sha(result_path), 'assets_verified': len(result['assets'])})
    write(STUDY / f'{name}_verification.json', {'status': 'passed', 'completed_training_count': len(validation),
        'total_training_process_seconds': sum(r['elapsed_seconds'] for r in validation), 'runs': validation,
        'historical_training_count': 134, 'previous_engineering_count': 5, 'new_API_calls': 0, 'test_access': False})
    files = []
    manifest_path = STUDY / f'{name}_asset_manifest.json'
    receipt_path = STUDY / f'{name}_archive_receipt.json'
    for path in sorted(STUDY.rglob('*')):
        if not path.is_file():
            continue
        if path in (manifest_path, receipt_path):
            continue
        if '/runs/' in path.as_posix() and not (path.parent / 'result.json').exists():
            continue
        if metadata_only and path.suffix not in ('.json', '.md', '.py'):
            continue
        files.append(path)
    script_names = ('run_direct_processing.py', 'run_direct_batch.py', 'confirm_direct_wave1.py', 'run_direct_own_ablation.py',
        'run_direct_mechanism_batch.py', 'check_direct_processing.py', 'diagnose_binary_support.py', 'audit_direct_mechanism.py', 'package_direct_study.py')
    files += [PROJECT / 'scripts' / n for n in script_names if (PROJECT / 'scripts' / n).exists()]
    manifest = {path.relative_to(PROJECT).as_posix(): {'sha256': sha(path), 'bytes': path.stat().st_size} for path in files}
    write(manifest_path, manifest)
    files.append(manifest_path)
    with tarfile.open(destination, 'w') as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(PROJECT).as_posix(), recursive=False)
    receipt = {'archive': str(destination), 'sha256': sha(destination), 'bytes': destination.stat().st_size,
        'members': len(files), 'training_runs': len(validation), 'metadata_only': metadata_only}
    write(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--archive-directory', help='Existing server directory with enough free space for the archive')
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError('Invalid archive name')
    main(args.name, args.metadata_only, args.archive_directory)
