"""Verify source-window metadata and CSV exports against remote hash receipts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'research_runs/selection'


def main():
    receipt=json.loads((ROOT/'data_asset_checksums.json').read_text())
    for name,expected in receipt.items():
        with (ROOT/name).open('rb') as f:
            actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:
            raise ValueError(f'Metadata mismatch: {name}')
    print(f'PASS: {len(receipt)} selection metadata/export hashes match remote receipts.')


if __name__=='__main__':
    main()
