"""Verify a disclosed parent, prepare only F/V; later C requires selection lock."""
import argparse
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_ts.auto_calibration import ForecastSamples, context_from_inputs, verify_time_order
from run_research import DLinear

STAGE = ROOT/'research_runs/agent_research/auto_calibration/20261021_r1'
PARENT = ROOT/'research_runs/ett_confirmation/ETTh1/runs/PARENT_P_s2026101901/last.pt'
ARCHIVE = ROOT/'research_runs/ett_confirmation/archives/ETTh1_results.tar.gz'
ARCHIVE_SHA = '79923933dd2649bca8c011fceb48afe0bd7a304bbd71320687865b4d4e811267'
RAW = ROOT/'data/ett_v1/raw/ETTh1.csv'
RAW_SHA = 'f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066'
COLUMNS = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
CONFIG = dict(input_length=96, prediction_length=96, channels=7, moving_average_kernel=25)
SPLITS = {'F': (8640, 9360), 'V': (9360, 10080), 'C': (10080, 11520)}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_prefix(path, count, columns=COLUMNS):
    # Do not request a numeric row beyond the explicitly authorized prefix.
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        if next(reader) != columns:
            raise ValueError('source columns changed')
        rows = list(itertools.islice(reader, count))
    if len(rows) != count or any(len(row) != len(columns) for row in rows):
        raise ValueError('prefix shape changed')
    values = np.array([[float(v) for v in row[1:]] for row in rows], dtype=np.float64)
    dates = np.array([row[0] for row in rows], dtype='datetime64[s]')
    if not np.isfinite(values).all() or not np.all(np.diff(dates) == np.timedelta64(3600, 's')):
        raise ValueError('invalid values or non-hourly source chronology')
    return values


def verified_parent():
    if sha(ARCHIVE) != ARCHIVE_SHA or sha(RAW) != RAW_SHA:
        raise ValueError('registered archive/source identity changed')
    suffix = 'runs/PARENT_P_s2026101901/last.pt'
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
        if len(members) != 1:
            raise ValueError('parent archive member is ambiguous')
        with archive.extractfile(members[0]) as stream:
            archived_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if sha(PARENT) != archived_sha:
        raise ValueError('local parent differs from verified archive')
    # This is the hash-verified project checkpoint, including historical RNGs.
    state = torch.load(PARENT, map_location='cpu', weights_only=False)
    if state['step'] != 3000:
        raise ValueError('only the fixed last checkpoint is allowed')
    model = DLinear(CONFIG)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, archived_sha


def reference_forward(model, x):
    """NumPy float64 reference for the inherited backbone, independent of torch pooling."""
    x = x.astype(np.float64)
    padded = np.pad(x, ((0, 0), (12, 12), (0, 0)), mode='edge')
    trend = np.lib.stride_tricks.sliding_window_view(padded, 25, axis=1).mean(axis=-1)
    state = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    return np.stack([(x[:, :, c]-trend[:, :, c]) @ state[f'seasonal.{c}.weight'].T +
                     state[f'seasonal.{c}.bias'] + trend[:, :, c] @ state[f'trend.{c}.weight'].T +
                     state[f'trend.{c}.bias'] for c in range(7)], axis=-1)


def build_samples(normalized, model, split):
    lo, hi = SPLITS[split]
    starts = np.arange(lo, hi-191, dtype=np.int64)
    predictions, contexts, truths = [], [], []
    with torch.no_grad():
        for first in range(0, len(starts), 256):
            block = normalized[starts[first:first+256, None] + np.arange(192)]
            x, y = block[:, :96], block[:, 96:]
            predictions.append(model(torch.from_numpy(x.copy())).numpy())
            contexts.append(context_from_inputs(x))
            truths.append(y)
    return ForecastSamples(np.concatenate(predictions), np.concatenate(contexts),
                           np.concatenate(truths), starts, starts+191)


def save_samples(path, samples):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **vars(samples))


def load_samples(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(ForecastSamples.__dataclass_fields__):
            raise ValueError('unexpected sample fields')
        return ForecastSamples(**{k: data[k].copy() for k in data.files})


def prepare(future=False):
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if future:
        lock = read(STAGE/'selection_lock.json')
        if lock['status'] != 'all_paths_complete_future_unscored':
            raise ValueError('future evaluation requires completed selection lock')
        existing = read(STAGE/'input_receipt.json')
    else:
        (STAGE/'inputs').mkdir()
    model, parent_sha = verified_parent()
    raw = load_prefix(RAW, 11520 if future else 10080)
    mean, std = raw[:8640].mean(axis=0), raw[:8640].std(axis=0)
    scale = np.where(std == 0, 1., std)
    normalized = ((raw-mean)/scale).astype(np.float32)
    probe = normalized[8640 + np.arange(192)][:96][None].copy()
    expected = reference_forward(model, probe)
    with torch.no_grad():
        actual = copy.deepcopy(model).double()(torch.from_numpy(probe.astype(np.float64))).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    result = {}
    for split in (('C',) if future else ('F', 'V')):
        item = build_samples(normalized, model, split)
        path = STAGE/'inputs'/f'{split}.npz'
        save_samples(path, item)
        result[split] = {'sha256': sha(path), 'windows': len(item.prediction),
                         'first_input_row': int(item.input_start[0]),
                         'last_label_row': int(item.label_end[-1]), 'bytes': path.stat().st_size}
    if not future:
        verify_time_order(load_samples(STAGE/'inputs/F.npz'), load_samples(STAGE/'inputs/V.npz'))
    elif parent_sha != existing['parent_sha256'] or mean.tolist() != existing['mean'] or scale.tolist() != existing['scale']:
        raise ValueError('parent or preprocessing differs from development')
    receipt = {'status': 'passed', 'source_sha256': RAW_SHA, 'parent_sha256': parent_sha,
               'parent_step': 3000, 'mean': mean.tolist(), 'scale': scale.tolist(),
               'normalization_fit_rows': [0, 8640], 'parsed_numeric_rows': [0, len(raw)],
               'splits': result, 'backbone_reference_max_error': float(np.max(np.abs(actual-expected))),
               'D_values_parsed': False, 'future_scored': False,
               'new_backbone_training': 0, 'calibrator_training': 0,
               'seconds': time.perf_counter()-started}
    write(STAGE/('future_input_receipt.json' if future else 'input_receipt.json'), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--future', action='store_true')
    prepare(parser.parse_args().future)
