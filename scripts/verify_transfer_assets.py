"""Verify transferred temporal-diagnosis and alignment assets without GPU libraries."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'research_runs/transfer'


def main():
    receipt=json.loads((ROOT/'asset_checksums.json').read_text())
    for name,expected in receipt.items():
        with (ROOT/name).open('rb') as handle: actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=expected: raise ValueError('Transfer asset mismatch: '+name)
    print(f'PASS: {len(receipt)} temporal/alignment assets match remote receipts.')


if __name__=='__main__': main()
