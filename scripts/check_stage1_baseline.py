"""Zero-training reference transform and deployment checks."""
import argparse
import importlib.util
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    spec = importlib.util.spec_from_file_location('baseline', root/'run_intervention_stage1_baseline.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    protocol = runner.read(root/'baseline_protocol.json')
    assert runner.sha(root/'run_intervention_stage1_baseline.py') == protocol['runner_sha256']
    recipe = protocol['source_half_recipe']
    x = torch.arange(3*32*15, dtype=torch.float64).reshape(3, 32, 15).square()/1000
    original = x.clone()
    actual = runner.transform(x, 'FIX', recipe)
    expected = x.clone()
    for target in recipe['changed_channels']:
        source = recipe['mapping'][target]
        expected[:, :, target] = x[:, -1:, target] + .5*(x[:, :, target]-x[:, -1:, target]) + .5*(x[:, :, source]-x[:, -1:, source])
    assert torch.equal(actual, expected) and torch.equal(x, original)
    assert torch.equal(actual[:, -1], x[:, -1])
    assert torch.equal(runner.transform(x, 'B0', recipe), x)
    starts = torch.tensor([0, 7, 60])
    data = torch.arange(100*15).reshape(100, 15)
    batched = data[starts[:, None]+torch.arange(40)]
    assert torch.equal(batched, torch.stack([data[int(start):int(start)+40] for start in starts]))
    for task, expected_hash in protocol['task_array_sha256'].items():
        assert runner.sha(root/'tasks'/task/'arrays.npz') == expected_hash
    assert runner.sha(protocol['frozen_core']) == protocol['frozen_core_sha256']
    print('PASS: frozen runner/core/task arrays; FIX formula, input immutability, anchor, B0 identity and window gather. No training.')


if __name__ == '__main__':
    main()
