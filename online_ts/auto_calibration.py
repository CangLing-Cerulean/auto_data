"""Train a small output calibrator with explicit, inspectable recipes.

The backbone is outside this module. A training call receives only its fitting
samples; validation/future truth is never an optimizer input.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WINDOWS = ("all", "recent_half", "recent_quarter")
LEARNING_RATES = (0.0001, 0.0005, 0.001, 0.005)
UPDATE_STEPS = (40, 80, 160)
LOSSES = ("mse", "huber")
WEIGHT_DECAYS = (0.0, 0.0001)
INITIALIZATIONS = ("identity", "incumbent")
BATCH_SIZE = 64
CONTEXT_SIZE = 3


@dataclass(frozen=True)
class Recipe:
    window: str = "all"
    learning_rate: float = 0.001
    steps: int = 160
    loss: str = "mse"
    weight_decay: float = 0.0001
    initialization: str = "identity"

    def __post_init__(self):
        choices = {"window": WINDOWS, "learning_rate": LEARNING_RATES,
                   "steps": UPDATE_STEPS, "loss": LOSSES,
                   "weight_decay": WEIGHT_DECAYS, "initialization": INITIALIZATIONS}
        for name, allowed in choices.items():
            value = getattr(self, name)
            if isinstance(value, bool) or value not in allowed:
                raise ValueError(f"unsupported {name}: {value!r}")
        if type(self.steps) is not int:
            raise ValueError("steps must be an integer")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("recipe must contain exactly the six declared fields")
        return cls(**value)

    def key(self):
        text = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class ForecastSamples:
    prediction: np.ndarray  # [sample, horizon, channel]
    context: np.ndarray  # [sample, channel, mean/last/std]
    truth: np.ndarray
    input_start: np.ndarray  # inclusive source rows
    label_end: np.ndarray  # inclusive source rows

    def __post_init__(self):
        p, c, y = self.prediction, self.context, self.truth
        if not all(isinstance(a, np.ndarray) for a in (p, c, y, self.input_start, self.label_end)):
            raise ValueError("sample fields must be numpy arrays")
        if p.ndim != 3 or min(p.shape) <= 0 or y.shape != p.shape:
            raise ValueError("prediction/truth must have identical nonempty [N,H,C] shape")
        if c.shape != (len(p), p.shape[2], CONTEXT_SIZE):
            raise ValueError("context must have exact [N,C,3] shape")
        if not all(a.dtype.kind in "fiu" and np.isfinite(a).all() for a in (p, c, y)):
            raise ValueError("nonfinite or nonnumeric samples")
        for name in ("input_start", "label_end"):
            a = getattr(self, name)
            if a.shape != (len(p),) or a.dtype.kind not in "iu":
                raise ValueError(f"{name} must be a row-aligned integer vector")
            if np.any(a < 0) or np.any(a[1:] <= a[:-1]):
                raise ValueError(f"{name} must be nonnegative and strictly increasing")
        if np.any(self.label_end <= self.input_start):
            raise ValueError("labels must end after input begins")

    def indices_for(self, window):
        if window not in WINDOWS:
            raise ValueError("unknown data window")
        fraction = {"all": 1.0, "recent_half": 0.5, "recent_quarter": 0.25}[window]
        count = math.ceil(len(self.prediction) * fraction)
        return np.arange(len(self.prediction) - count, len(self.prediction))


def verify_time_order(fitting, validation):
    if fitting.prediction.shape[1:] != validation.prediction.shape[1:]:
        raise ValueError("forecast dimensions changed across splits")
    if fitting.label_end[-1] >= validation.input_start[0]:
        raise ValueError("fitting labels overlap validation inputs")


def context_from_inputs(inputs):
    """All features use the observation window available at forecast issue time."""
    x = np.asarray(inputs)
    if x.ndim != 3 or min(x.shape) <= 0 or not np.isfinite(x).all():
        raise ValueError("inputs must be finite nonempty [N,L,C]")
    with np.errstate(over="raise", invalid="raise"):
        result = np.stack((x.mean(axis=1), x[:, -1], x.std(axis=1)), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("context overflow")
    return result


class GatedLinearCalibrator(nn.Module):
    """Independently implemented gated residual head; COSA structural baseline.

    Context is OUR fixed per-channel input mean/last/std, not the official COSA
    context buffer. This class alone is not an official COSA reproduction.
    """
    def __init__(self, horizon, channels, seed=2026102101):
        super().__init__()
        if type(horizon) is not int or type(channels) is not int or min(horizon, channels) <= 0:
            raise ValueError("positive integer horizon and channels required")
        self.horizon, self.channels = horizon, channels
        self.weight = nn.Parameter(torch.empty(channels, horizon + CONTEXT_SIZE, horizon))
        self.bias = nn.Parameter(torch.zeros(channels, horizon))
        self.gate = nn.Parameter(torch.zeros(channels))
        # Preserve process RNG state and initialize each channel as one Linear.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for channel in range(channels):
                nn.init.xavier_uniform_(self.weight[channel].T, gain=0.1)

    def forward(self, prediction, context):
        if prediction.ndim != 3 or prediction.shape[1:] != (self.horizon, self.channels):
            raise ValueError("prediction shape changed")
        if context.shape != (len(prediction), self.channels, CONTEXT_SIZE):
            raise ValueError("context shape changed")
        z = torch.cat((prediction.transpose(1, 2), context), dim=-1)
        correction = torch.einsum("nci,cio->nco", z, self.weight)
        # An ungated zero-initialized offset remains learnable even when z=0.
        correction = correction * torch.tanh(self.gate)[None, :, None] + self.bias
        return prediction + correction.transpose(1, 2)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_candidate(fitting, recipe, *, seed=2026102101, incumbent=None):
    """Fit only F, with fresh Adam state even for an incumbent warm start.

    Model weights are copied before optimization. A recipe's historical parent
    checkpoint and the cost that created it must be accounted by the caller.
    """
    if not isinstance(recipe, Recipe) or not isinstance(fitting, ForecastSamples):
        raise TypeError("validated samples and Recipe required")
    start = time.perf_counter()
    horizon, channels = fitting.prediction.shape[1:]
    if recipe.initialization == "incumbent":
        if incumbent is None or (incumbent.horizon, incumbent.channels) != (horizon, channels):
            raise ValueError("a dimension-matched incumbent is required")
        model = copy.deepcopy(incumbent).cpu()
    else:
        model = GatedLinearCalibrator(horizon, channels, seed)
    before = state_digest(model)
    model.train()
    tensors = [torch.as_tensor(np.array(a, dtype=np.float32, copy=True))
               for a in (fitting.prediction, fitting.context, fitting.truth)]
    if not all(torch.isfinite(a).all() for a in tensors):
        raise ValueError("float32 sample conversion overflow")
    ids = fitting.indices_for(recipe.window)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe.learning_rate,
                                 weight_decay=recipe.weight_decay,
                                 betas=(0.9, 0.999), eps=1e-8, foreach=False)
    losses = []
    for _ in range(recipe.steps):
        chosen = torch.as_tensor(rng.choice(ids, size=BATCH_SIZE, replace=True))
        p, c, y = (a[chosen] for a in tensors)
        optimizer.zero_grad(set_to_none=True)
        calibrated = model(p, c)
        loss = F.mse_loss(calibrated, y) if recipe.loss == "mse" else F.huber_loss(calibrated, y, delta=1.0)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite or missing gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite adapter parameter")
        losses.append(float(loss.detach()))
    model.eval()
    return model, {"recipe": asdict(recipe), "seed": seed, "steps": recipe.steps,
                   "training_windows": len(ids), "first_training_index": int(ids[0]),
                   "last_training_label_row": int(fitting.label_end[ids[-1]]),
                   "initial_state_sha256": before, "final_state_sha256": state_digest(model),
                   "train_loss_first": losses[0], "train_loss_last": losses[-1],
                   "seconds": time.perf_counter() - start}


def evaluate(model, samples):
    """Return actual held-out metrics; do not fit any parameter here."""
    before = state_digest(model)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples.prediction), 256):
            p = torch.as_tensor(samples.prediction[start:start+256], dtype=torch.float32)
            c = torch.as_tensor(samples.context[start:start+256], dtype=torch.float32)
            predictions.append(model(p, c).cpu().numpy())
    prediction = np.concatenate(predictions)
    if not np.isfinite(prediction).all() or state_digest(model) != before:
        raise ValueError("evaluation changed model or produced nonfinite predictions")
    error = prediction.astype(np.float64) - samples.truth.astype(np.float64)
    with np.errstate(over="raise", invalid="raise"):
        mse = float(np.mean(error**2))
        mae = float(np.mean(np.abs(error)))
        channel_mse = np.mean(error**2, axis=(0, 1)).tolist()
        horizon_mse = np.mean(error**2, axis=(0, 2)).tolist()
    return {"mse": mse, "mae": mae, "channel_mse": channel_mse,
            "horizon_mse": horizon_mse, "mean_error": float(np.mean(error)),
            "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest()}, prediction
