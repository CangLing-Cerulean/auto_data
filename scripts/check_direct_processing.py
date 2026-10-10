"""Meaningful preprocessing checks on actual T records, without training."""
import csv

import numpy as np
import torch

from run_direct_processing import BASE, CHANGED, MAPPING, STUDY, load, sha, shifts, transform, write


def main():
    audit_path = STUDY / 'component_validation.json'
    if audit_path.exists():
        raise ValueError('Validation already recorded')
    audit = load(STUDY / 'runs/DP01_o60_baseline_s2026/data_audit.json')
    path = BASE / 'data/metropt3_v1/train.csv'
    raw = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(2, 17), max_rows=4096)
    values = ((raw - np.asarray(audit['mean'])) / np.asarray(audit['scale'])).astype(np.float32)
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        next(reader)
        ids = np.array([int(next(reader)[0]) for _ in range(4096)], dtype=np.int64)
    starts = np.arange(0, 4000, 29)
    x = torch.from_numpy(values[starts[:, None] + np.arange(32)])
    original = x.clone()
    source_ids = torch.from_numpy(ids[starts])
    assert torch.equal(transform(x, source_ids, 'baseline'), x)
    assert len(CHANGED) == 6
    assert len(torch.unique(shifts(source_ids))) > 1
    for policy in ('source_prior', 'source_half', 'source_half_scramble'):
        actual = transform(x, source_ids, policy)
        assert actual.shape == x.shape and torch.isfinite(actual).all()
        assert torch.equal(actual[:, :, MAPPING == np.arange(15)], x[:, :, MAPPING == np.arange(15)])
        assert torch.equal(actual[:, -1], x[:, -1])
    sources = x[:, :, MAPPING.tolist()]
    expected = sources - sources[:, -1:] + x[:, -1:]
    expected[:, :, MAPPING == np.arange(15)] = x[:, :, MAPPING == np.arange(15)]
    assert torch.allclose(transform(x, source_ids, 'source_prior'), expected, atol=2e-6, rtol=2e-6)
    half = (x + expected) / 2
    assert torch.allclose(transform(x, source_ids, 'source_half'), half, atol=2e-6, rtol=2e-6)
    rotation = (torch.arange(32)[None, :] + shifts(source_ids)[:, None]) % 32
    scrambled = sources.gather(1, rotation[:, :, None].expand(-1, -1, 15))
    expected_control = x[:, -1:] + .5 * (x - x[:, -1:]) + .5 * (scrambled - scrambled[:, -1:])
    expected_control[:, :, MAPPING == np.arange(15)] = x[:, :, MAPPING == np.arange(15)]
    assert torch.equal(transform(x, source_ids, 'source_half_scramble'), expected_control)
    assert torch.equal(original, x)
    write(audit_path, {'status': 'passed', 'scope': 'actual_training_records_only_no_training', 'checks': [
        'baseline_identity', 'six_changed_channels', 'window_dependent_rotation', 'unchanged_channels_exact',
        'last_observation_preserved_exactly', 'legacy_source_formula', 'half_source_formula', 'scramble_formula',
        'input_not_mutated', 'finite_output_and_shape'], 'real_input_windows': len(starts),
        'source_train_sha256': sha(path), 'code_sha256': sha(__file__), 'training_calls': 0, 'generated_sensor_records': 0})
    print('PASS: 10 preprocessing checks on actual T records; no training')


if __name__ == '__main__':
    main()
