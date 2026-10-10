"""Training-only diagnostics. Never loads test or ground_truth records."""
import csv
from pathlib import Path

import numpy as np

from ..config import GateError
from .common import ROOT, STUDY, file_hash, read_json, write_json

FEATURES = ("input_variation", "analog_mean", "analog_std", "digital_transitions",
            "elapsed_days", "month", "state", "group_size")
DATASET = ROOT / "data/metropt3_v1"


def valid_starts(times, length=40, gap=30):
    delta = np.diff(times).astype("timedelta64[s]").astype(np.int64)
    if np.any(delta <= 0):
        raise GateError("TIMESTAMPS_NOT_INCREASING")
    bad = np.r_[0, np.cumsum(delta > gap)]
    starts = np.arange(max(0, len(times) - length + 1))
    return starts[(bad[starts + length - 1] - bad[starts]) == 0]


def prepare_task(destination):
    """Read only immutable T; D is referenced by hash, not profiled."""
    destination = Path(destination).resolve()
    if not destination.is_relative_to((STUDY / "tasks").resolve()) or destination.exists():
        raise GateError("NEW_TASK_DIRECTORY_REQUIRED")
    manifest_path = DATASET / "manifest.json"
    manifest = read_json(manifest_path)
    protocol_path = ROOT / "research_runs/protocol.json"
    protocol = read_json(protocol_path)
    expected = next(s for s in manifest["splits"] if s["name"] == "train")
    if file_hash(DATASET / "train.csv") != expected["sha256"]:
        raise GateError("TRAIN_HASH")
    raw = np.loadtxt(DATASET / "train.csv", delimiter=",", skiprows=1, usecols=range(2, 17))
    with (DATASET / "train.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [(int(row[0]), row[1]) for row in reader]
    if header != manifest["columns_raw"] or len(raw) != expected["rows"] or not np.isfinite(raw).all():
        raise GateError("TRAIN_SCHEMA")
    source_ids = np.array([r[0] for r in rows], dtype=np.int64)
    times = np.array([r[1] for r in rows], dtype="datetime64[s]")
    starts = valid_starts(times, protocol["input_length"] + protocol["prediction_length"], protocol["max_gap_seconds"])
    mean, std = raw.mean(0), raw.std(0)
    scale = np.where(std == 0, 1, std)
    values = (raw - mean) / scale
    length = protocol["input_length"]
    changes = np.abs(np.diff(values, axis=0)).mean(1)
    prefix = np.r_[0, np.cumsum(changes)]
    variation = (prefix[starts + length - 1] - prefix[starts]) / (length - 1)
    analog_mean = np.r_[0, np.cumsum(values[:, :7].mean(1))]
    analog_square = np.r_[0, np.cumsum(np.square(values[:, :7]).mean(1))]
    av = (analog_mean[starts + length] - analog_mean[starts]) / length
    variance = (analog_square[starts + length] - analog_square[starts]) / length - av ** 2
    if not np.isin(raw[:, 7:], (0, 1)).all():
        raise GateError("DIGITAL_SCHEMA")
    transitions = np.r_[0, np.cumsum(np.any(np.diff(raw[:, 7:], axis=0) != 0, axis=1))]
    months = times[starts + length].astype("datetime64[M]").astype(str)
    state = (raw[starts + length - 1, 7:].astype(np.int64) * (1 << np.arange(8))).sum(1)
    group_names, groups = np.unique(np.array([f"{m}|state={s}" for m, s in zip(months, state)]), return_inverse=True)
    features = {"input_variation": variation, "analog_mean": av,
                "analog_std": np.sqrt(np.maximum(variance, 0)),
                "digital_transitions": transitions[starts + length - 1] - transitions[starts],
                "elapsed_days": (times[starts] - times[0]).astype("timedelta64[s]").astype(float) / 86400,
                "month": times[starts + length].astype("datetime64[M]").astype(int) % 12 + 1,
                "state": state, "group_size": np.bincount(groups)[groups]}
    mapping = {"start_row": starts, "source_index_start": source_ids[starts],
               "source_index_target_start": source_ids[starts + length],
               "source_index_end_inclusive": source_ids[starts + length + protocol["prediction_length"] - 1],
               "groups": groups, "group_names": group_names}
    historical = []
    paths = list((ROOT / "research_runs/cluster_selection").glob("mask_*_target.npz"))
    paths += list((ROOT / "research_runs/results").glob("*/windows.npz"))
    paths += list((ROOT / "research_runs/results").glob("*/sampling_weights.npz"))
    seen = set()
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as old:
            if "retained" in old.files:
                old_retained = old["retained"]
            elif "probability" in old.files:
                old_retained = old["probability"] > 0
            else:
                continue
            if old_retained.all():
                continue
            if "start_row" in old.files:
                old_starts = old["start_row"]
            elif "train_start" in old.files:
                old_starts = old["train_start"]
            elif path.name == "sampling_weights.npz" and len(old_retained) == len(starts):
                # Original full-partition sampler is bound to this frozen table.
                table = ROOT / "research_runs/selection/window_table.npz"
                with np.load(table, allow_pickle=False) as base:
                    old_starts = base["start_row_zero_based"]
            else:
                raise GateError("HISTORICAL_MASK_MAPPING_UNAVAILABLE")
            indices = np.searchsorted(starts, old_starts)
            if (len(old_retained) != len(old_starts) or np.any(indices >= len(starts))
                    or not np.array_equal(starts[indices], old_starts)):
                raise GateError("HISTORICAL_WINDOW_MAPPING")
            import hashlib
            membership_hash = hashlib.sha256(old_starts.tobytes() + np.packbits(old_retained).tobytes()).hexdigest()
            if membership_hash in seen:
                continue
            seen.add(membership_hash)
            domain, removed = np.zeros(len(starts), bool), np.zeros(len(starts), bool)
            domain[indices] = True
            removed[indices] = ~old_retained
        index = len(historical)
        mapping[f"historical_domain_{index}"] = domain
        mapping[f"historical_removed_{index}"] = removed
        historical.append({"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path)})
    destination.mkdir(parents=True)
    np.savez_compressed(destination / "features.npz", **features, **mapping)
    task = {"task_id": "metropt3_development_v1", "dataset_id": manifest["dataset_id"],
            "diagnostic_code_sha256": file_hash(Path(__file__)),
            "manifest_sha256": file_hash(manifest_path), "protocol_sha256": file_hash(protocol_path),
            "protocol": protocol, "evaluation_scope": "development_only",
            "train_sha256": expected["sha256"],
            "development_sha256": next(s["sha256"] for s in manifest["splits"] if s["name"] == "validation"),
            "features_path": (destination / "features.npz").relative_to(STUDY).as_posix(),
            "features_sha256": file_hash(destination / "features.npz"), "feature_names": list(FEATURES),
            "rows": len(raw), "legal_windows": len(starts), "sensor_columns": manifest["sensor_columns"],
            "mean": mean.tolist(), "scale": scale.tolist(), "historical_masks": historical,
            "roles": {"T": "original train.csv", "D": "original validation.csv; metrics through trainer only",
                      "E": "absent; old test prohibited"},
            "mapping": "start_row s -> unchanged train rows [s,s+40); input [s,s+32), targets [s+32,s+40)"}
    write_json(destination / "task.json", task)
    return task


class TrainingData:
    def __init__(self, task):
        self.task = task
        path = (STUDY / task["features_path"]).resolve()
        if not path.is_relative_to((STUDY / "tasks").resolve()) or file_hash(path) != task["features_sha256"]:
            raise GateError("FEATURE_ASSET_HASH")
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {k: archive[k].copy() for k in archive.files}
        self.starts = self.arrays["start_row"]
        self.features = {k: self.arrays[k] for k in FEATURES}
        self.groups = self.arrays["groups"]

    def profile(self):
        return {"scope": "training_only", "windows": len(self.starts), "mapping": self.task["mapping"],
                "features": {k: {"min": float(v.min()), "max": float(v.max()),
                    "quantiles": np.quantile(v, [.1, .25, .5, .75, .9]).tolist()} for k, v in self.features.items()},
                "groups": [{"name": str(n), "windows": int(np.sum(self.groups == i))}
                           for i, n in enumerate(self.arrays["group_names"])],
                "interpretation": "input diagnostics; no independently verified fault labels"}
