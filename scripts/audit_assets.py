"""Create remote artifact receipts, or verify the fetched local copies."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    count=0
    for run in sorted((ROOT/'research_runs/results').iterdir()):
        if not (run/'result.json').exists():
            continue
        receipt=run/'asset_checksums.json'
        if args.write:
            hashes={p.name:digest(p) for p in run.iterdir() if p.is_file() and p!=receipt and p.suffix!='.tmp'}
            receipt.write_text(json.dumps(hashes,indent=2)+'\n')
        else:
            hashes=json.loads(receipt.read_text())
            for name,expected in hashes.items():
                if digest(run/name)!=expected:
                    raise ValueError(f'Asset mismatch: {run.name}/{name}')
        count+=1
    print(f'PASS: {count} experiment artifact receipts '+('written.' if args.write else 'verified.'))


if __name__=='__main__':
    main()
