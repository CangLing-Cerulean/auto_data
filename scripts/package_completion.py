"""Package only completed new study runs; never read base test observations."""
import argparse
import json
import tarfile
from pathlib import Path
from run_research import ROOT, digest, write_json

FOLDER = ROOT / 'research_runs/completion_study'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name', required=True); parser.add_argument('--runs', nargs='+', required=True)
    args = parser.parse_args(); assert Path(args.name).name == args.name
    receipt = {}; runs = []
    for name in args.runs:
        assert Path(name).name == name
        run = ROOT / 'research_runs/results' / name
        assert json.loads((run / 'status.json').read_text())['phase'] == 'completed'
        files = {path.name: digest(path) for path in run.iterdir() if path.is_file() and path.name != 'asset_checksums.json'}
        receipt_path = run / 'asset_checksums.json'
        if receipt_path.exists(): assert json.loads(receipt_path.read_text()) == files
        else: write_json(receipt_path, files)
        for filename, sha in files.items(): receipt[(run / filename).relative_to(ROOT).as_posix()] = sha
        receipt[receipt_path.relative_to(ROOT).as_posix()] = digest(receipt_path)
        runs.append(name)
    output = FOLDER / (args.name + '_receipt.json'); assert not output.exists()
    write_json(output, {'runs': runs, 'files': receipt})
    archive = FOLDER / (args.name + '.tar.gz'); assert not archive.exists()
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for relative in receipt: tar.add(ROOT / relative, arcname=relative, recursive=False)
        tar.add(output, arcname=output.relative_to(ROOT).as_posix(), recursive=False)
    print(json.dumps({'archive': str(archive), 'archive_sha256': digest(archive), 'runs': len(runs), 'files': len(receipt), 'bytes': archive.stat().st_size}), flush=True)


if __name__ == '__main__': main()
