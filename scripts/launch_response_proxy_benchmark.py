"""Launch the registered engineering matrix with file-backed output, exactly once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
out = Path(parser.parse_args().output).resolve()
plan = read(out / 'plan.json')
for name, digest in plan['code_sha256'].items():
    assert sha(out / name) == digest, name
assert read(out / 'source_sync_receipt.json')['status'] == 'exact_commit_and_tree_verified'
numeric = read(out / 'numerical' / 'result.json')
assert numeric['status'] == 'PASS' and numeric['plan_sha256'] == sha(out / 'plan.json')
assert not (out / 'runs').exists() and not (out / 'benchmark.lock').exists()
control = out / 'control'
control.mkdir()
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
with (control / 'run.log').open('xb') as log:
    process = subprocess.Popen([sys.executable, '-B', str(out / 'verify_response_proxy.py'),
        'benchmark', '--output', str(out)], stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=out)
with (control / 'launch.json').open('x', encoding='utf-8') as stream:
    json.dump(dict(pid=process.pid, launched_unix=time.time(), plan_sha256=sha(out / 'plan.json'),
                   output_transport='regular_file'), stream, indent=2)
print(json.dumps(dict(pid=process.pid, status='launched', registered=len(plan['jobs']))))
