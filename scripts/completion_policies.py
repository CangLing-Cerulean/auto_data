"""Deterministic train-only policy arithmetic, independent of the training loop."""
import numpy as np

MEMBER_SEEDS = (20261021, 20261022, 20261023)
SHUFFLE_SEED = 20261021


def rank_weights(values):
    """Bounded decreasing weights with exactly unit mean, including ties."""
    values = np.asarray(values)
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float32)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts + 1) / 2
    weights = 1.5 - average_rank[inverse] / (len(values) - 1)
    assert abs(weights.mean() - 1) < 1e-12
    return weights.astype(np.float32)


def oof_weights(errors, scored, cells):
    n = len(errors)
    window = np.ones(n, dtype=np.float32)
    target = np.ones(errors.shape, dtype=np.float32)
    permutation = np.arange(n)
    rng = np.random.default_rng(SHUFFLE_SEED)
    for cell in np.unique(cells):
        members = np.flatnonzero((cells == cell) & scored)
        if not len(members):
            continue
        window[members] = rank_weights(errors[members].mean((1, 2), dtype=np.float64))
        for horizon in range(8):
            for channel in range(15):
                target[members, horizon, channel] = rank_weights(errors[members, horizon, channel])
        permutation[members] = rng.permutation(members)
        assert np.allclose(target[members].mean(0, dtype=np.float64), 1, atol=1e-7, rtol=0)
    assert np.all((window >= .5) & (window <= 1.5))
    assert np.all((target >= .5) & (target <= 1.5))
    assert np.all(window[~scored] == 1) and np.all(target[~scored] == 1)
    return window, target, permutation


def protected_windows(raw, starts):
    changed = np.any(np.diff(raw[:, 7:], axis=0) != 0, axis=1)
    prefix = np.r_[0, np.cumsum(changed)]
    return prefix[starts + 31] > prefix[starts]


def input_signatures(normalized, starts):
    source = normalized.astype(np.float64)
    features = np.empty((len(starts), 30), dtype=np.float64)
    for lo in range(0, len(starts), 8192):
        batch = source[starts[lo:lo + 8192, None] + np.arange(32)]
        features[lo:lo + len(batch), :15] = batch.mean(1)
        features[lo:lo + len(batch), 15:] = batch.std(1)
    center, scale = features.mean(0), features.std(0)
    scale = np.where(scale == 0, 1, scale)
    return ((features - center) / scale).astype(np.float32), center, scale


def quad_groups(cells, protected, seed):
    groups = []
    for cell in np.unique(cells):
        members = np.flatnonzero((cells == cell) & ~protected)
        members = members[seed % 4:]
        groups.append(members[:len(members) // 4 * 4].reshape(-1, 4))
    return np.concatenate(groups) if groups else np.empty((0, 4), dtype=np.int64)


def coreset_probability(features, quads, n, random_seed=None):
    if random_seed is None:
        representation = features[quads].astype(np.float64)
        distance = ((representation - representation.mean(1, keepdims=True)) ** 2).sum(2)
        position = distance.argmin(1)
    else:
        position = np.random.default_rng(random_seed).integers(4, size=len(quads))
    dropped = quads[np.arange(len(quads)), position]
    q = np.full(n, 1 / n, dtype=np.float64)
    q[quads.ravel()] = 4 / (3 * n)
    q[dropped] = 0
    assert abs(q.sum() - 1) < 1e-12
    assert np.allclose(q[quads].sum(1), 4 / n, atol=1e-15, rtol=0)
    return q, dropped


def shuffled_weights(weights, cells):
    result = weights.copy()
    rng = np.random.default_rng(SHUFFLE_SEED)
    permutation = np.arange(len(weights))
    for cell in np.unique(cells):
        members = np.flatnonzero(cells == cell)
        permutation[members] = rng.permutation(members)
    result[:] = weights[permutation]
    return result, permutation


def quality_profile(raw, names):
    rows = []
    for column, name in enumerate(names):
        values = raw[:, column]
        edges = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1, len(values)]
        longest = int(np.diff(edges).argmax())
        row = {'sensor': name, 'nonfinite': int((~np.isfinite(values)).sum()),
            'min': float(values.min()), 'max': float(values.max()),
            'longest_constant_rows': int(edges[longest + 1] - edges[longest]),
            'longest_constant_source_rows': [int(edges[longest]), int(edges[longest + 1])],
            'constant_value': float(values[edges[longest]])}
        if column >= 7:
            unique, counts = np.unique(values, return_counts=True)
            row.update(observed_values=unique.tolist(), counts=counts.tolist(), outside_binary_count=int((~np.isin(values, [0, 1])).sum()))
        rows.append(row)
    return {'source': 'current train prefix only', 'rows': len(raw), 'sensors': rows,
        'confirmed_measurement_faults': [], 'conclusion': 'Ranges and constant segments are diagnostic only; no independently verified sensor-error labels are available.'}


def verify_arithmetic():
    # Pure arithmetic checks; these arrays are never written as sensor observations.
    for values in ([0], [1, 1, 1], [1, 1, 2, 3], [0, 1, 2, 3]):
        weights = rank_weights(values)
        assert abs(weights.mean(dtype=np.float64) - 1) < 1e-7
        assert np.all(np.diff(weights) <= 0)
    features = np.arange(24, dtype=np.float64).reshape(8, 3)
    quads = np.arange(8).reshape(2, 4)
    q, removed = coreset_probability(features, quads, 8)
    assert len(removed) == 2 and abs(q.sum() - 1) < 1e-12
    return True
