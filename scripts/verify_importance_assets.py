"""Portable verification of research and diagnostic asset backups."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'research_runs/importance'
receipts=json.loads((folder/'asset_checksums.json').read_text())
for name,expected in receipts.items():
    with (folder/name).open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==expected,name
print(f'PASS: {len(receipts)} importance study assets verified.')
