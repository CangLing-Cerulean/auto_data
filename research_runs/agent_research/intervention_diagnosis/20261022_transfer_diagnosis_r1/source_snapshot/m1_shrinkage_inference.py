"""Local CPU inference with the original float32 branches and scoring rule."""
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select(curves):
    return min(sorted(curves, key=int), key=lambda t: curves[t]['B'])


def loss(prediction, target):
    return float(np.square((prediction - target).astype(np.float64)).mean())


def output_prediction(value, values, starts):
    outputs, targets = [], []
    for offset in range(0, len(starts), 2048):
        batch = values[starts[offset:offset + 2048, None] + torch.arange(40)]
        x = batch[:, :32].transpose(1, 2)
        trend = F.avg_pool1d(F.pad(x, (12, 12), mode='replicate'), 25, stride=1)
        outputs.append(torch.stack([F.linear((x - trend)[:, c], value[c, :32].T.contiguous(), value[c, 32]) +
            F.linear(trend[:, c], value[c, 33:65].T.contiguous(), value[c, 65]) for c in range(8)], -1).numpy())
        targets.append(batch[:, 32:].numpy())
    return np.concatenate(outputs), np.concatenate(targets)
