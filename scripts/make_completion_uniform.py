"""Export a uniform comparison policy from an audited, train-only policy source."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from run_research import ROOT, digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True); parser.add_argument('--source', required=True)
    args = parser.parse_args(); assert all(Path(v).name == v for v in (args.run_id, args.source))
    run = ROOT / 'research_runs/results' / args.run_id; assert not run.exists(); run.mkdir()
    begun = time.monotonic()
    source = ROOT / 'research_runs/results' / args.source
    assert json.loads((source / 'status.json').read_text())['phase'] == 'completed'
    source_identity = json.loads((source / 'identity.json').read_text())
    source_file = source / 'policy_oof_window.npz'
    with np.load(source_file) as saved:
        starts = saved['train_start'].copy(); cells = saved['cell'].copy()
    identity = {'run_type': 'fixed_uniform_policy_export', 'origin': source_identity['origin'],
        'source_run': args.source, 'source_sha256': digest(source_file),
        'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'), 'code_sha256': digest(Path(__file__))}
    write_json(run / 'identity.json', identity)
    (run / 'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    output = run / 'policy_uniform.npz'
    np.savez_compressed(output, train_start=starts, cell=cells, probability=np.full(len(starts), 1 / len(starts)),
        loss_weight=np.ones(len(starts), dtype=np.float32), expected_loss_weight=1.)
    write_json(run / 'result.json', {'run_id': args.run_id, 'identity': identity,
        'policies': [{'policy': 'uniform', 'sha256': digest(output)}], 'elapsed_seconds': time.monotonic() - begun})
    write_json(run / 'status.json', {'phase': 'completed', 'elapsed_seconds': time.monotonic() - begun})


if __name__ == '__main__': main()
