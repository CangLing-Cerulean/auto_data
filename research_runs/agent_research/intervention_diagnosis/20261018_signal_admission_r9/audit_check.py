"""Independent read-only r9 engineering audit; never imports production helpers.

gate: standard-library identity/AST checks only, no numerical data parsing.
full: reads explicitly authorized A/B arrays and saved parent parameters only.
"""
import argparse
import ast
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "research_runs/agent_research/intervention_diagnosis"
DEV = BASE / "20261013_response_proxy_development_r1/archives/verified"
INFO = BASE / "20261014_sampling_weight_information_r1/archives/unpacked"
ETT = ROOT / "research_runs/ett_confirmation"
CHECKS = []


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(name, condition, **details):
    CHECKS.append(dict(name=name, passed=bool(condition), **details))
    if not condition:
        raise AssertionError(name)


def close(name, actual, expected, rtol=2e-9, atol=2e-11):
    actual, expected = np.asarray(actual), np.asarray(expected)
    same = actual.shape == expected.shape
    gap = float(np.max(np.abs(actual-expected))) if same and actual.size else 0.
    check(name, same and bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
          shape=list(actual.shape), expected_shape=list(expected.shape), max_absolute_gap=gap,
          rtol=rtol, atol=atol)


def gate():
    lock = read(HERE / "execution_lock.json")
    check("task_parent_counts", len(lock["tasks"]) == 5 and sum(len(t["parents"]) for t in lock["tasks"]) == 15)
    check("input_count", len(lock["inputs"]) == 42, actual=len(lock["inputs"]))
    for name, expected in {**lock["inputs"], **lock["scripts"]}.items():
        path = ROOT / name
        check("sha256:"+name, digest(path) == expected)
    check("protocol_sha256", digest(HERE / "PROTOCOL.md") == lock["protocol_sha256"])
    source = (ROOT / "scripts/parent_directional_measurements.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    check("production_compiles", isinstance(tree, ast.Module))
    # Concrete boundary assertions supplement the manual static review.
    selected = [n.slice.value for n in ast.walk(tree) if isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == "z"
                and isinstance(n.slice, ast.Constant)]
    check("npz_member_boundary", set(selected) == {"A_values", "B_values", "A_starts", "B_starts", "A_cycles", "theta"}, members=sorted(set(selected)))
    check("csv_prefix_bound", "itertools.islice(reader, 10080)" in source)
    check("checkpoint_weights_only", "weights_only=True" in source)
    check("no_new_training_or_llm", lock["full_trainings"] == 0 and lock["LLM_calls"] == 0)
    for task in lock["tasks"]:
        check("even_horizon:"+task["task"], task["horizon"] % 2 == 0)
    # The ETT boundary retains 4224/4225 windows, so 3:1 per-window weights
    # intentionally yield a group mass different from exactly .75.
    counts = [sum(i+96 < 4320 for i in range(8449)), sum(i+96 >= 4320 for i in range(8449))]
    mass = 3*counts[0]/(3*counts[0]+counts[1])
    check("ett_per_window_mass_gate", counts == [4224, 4225] and mass != .75,
          counts=counts, FAVOR_EARLY_group_mass=mass)
    return lock


def load_ab(task):
    """Independent loaders: compressed archives read selected A/B members only."""
    name = task["task"]
    if task["family"] == "Hydraulic":
        with np.load(DEV / "prepared" / (name+"_ABC.npz"), allow_pickle=False) as archive:
            a = archive["A_values"].copy()
            b = archive["B_values"].copy()
            ai = archive["A_starts"].copy()
            bi = archive["B_starts"].copy()
            cycles = archive["A_cycles"].copy()
        plan_task = next(t for t in read(BASE / "20261014_weighted_response_r1/plan.json")["tasks"] if t["id"] == name)
        low, high = plan_task["partitions"]["A"]
        check("hydraulic_A_cycles:"+name, int(cycles.min()) == low and int(cycles.max()) == high-1)
        first = cycles[ai] < (low+high)/2
        for label, starts, values in [("A", ai, a), ("B", bi, b)]:
            check("window_bounds:"+name+label, starts.min() >= 0 and starts.max()+40 <= len(values))
            # Original Hydraulic cycles contain 60 rows; every legal 32->8
            # window stays within its cycle, giving 21 starts per cycle.
            expected_starts = (np.arange(0, len(values), 60)[:, None]+np.arange(21)).reshape(-1)
            close("complete_cycle_windows:"+name+label, starts, expected_starts, rtol=0, atol=0)
        check("hydraulic_window_counts:"+name, len(ai) == 308*21 and len(bi) == 77*21,
              A=len(ai), B=len(bi))
    else:
        with (ROOT / "data/ett_v1/raw" / (name+".csv")).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            check("csv_header:"+name, next(reader) == ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"])
            prefix = list(itertools.islice(reader, 10080))
        check("parsed_row_count:"+name, len(prefix) == 10080)
        raw = np.asarray([[float(x) for x in row[1:]] for row in prefix], dtype=np.float64)
        timestamps = np.asarray([row[0] for row in prefix], dtype="datetime64[s]")
        check("hourly_prefix:"+name, np.all(np.diff(timestamps) == np.timedelta64(1, "h")))
        mean = np.mean(raw[:8640], axis=0)
        scale = np.std(raw[:8640], axis=0)
        scale[scale == 0] = 1
        preprocessing = read(ETT / name / "preprocessing.json")
        close("normalization_mean:"+name, mean, preprocessing["mean"], rtol=0, atol=0)
        close("normalization_scale:"+name, scale, preprocessing["scale"], rtol=0, atol=0)
        # Preserve the original prepared float32 rounding before float64 audit.
        a = ((raw[:8640]-mean)/scale).astype(np.float32)
        b = ((raw[8640:10080]-mean)/scale).astype(np.float32)
        ai = np.arange(len(a)-191)
        bi = np.arange(len(b)-191)
        first = ai < 4320-96
        check("ett_window_counts:"+name, [len(ai), len(bi)] == [8449, 1249])
    check("finite_AB:"+name, np.isfinite(a).all() and np.isfinite(b).all())
    return a, b, ai, bi, first


def load_theta(task, parent):
    name = task["task"]
    if task["family"] == "Hydraulic":
        metadata = read(INFO / (name+"_diagnostic.json"))
        anchor = metadata["anchors"].index(dict(seed=parent["seed"], step=parent["step"]))
        with np.load(INFO / "statistics" / (name+".npz"), allow_pickle=False) as archive:
            return archive["theta"][anchor].astype(np.float64)
    saved = torch.load(ETT / name / "runs" / f"PARENT_P_s{parent['seed']}" / "best.pt", weights_only=True, map_location="cpu")
    check(f"parent_step:{name}:{parent['seed']}", saved["step"] == parent["step"])
    model = saved["model"]
    channels = []
    for channel in range(7):
        blocks = []
        for component in ("seasonal", "trend"):
            blocks += [model[f"{component}.{channel}.weight"].double().numpy().T,
                       model[f"{component}.{channel}.bias"].double().numpy().reshape(1, -1)]
        channels.append(np.vstack(blocks))
    return np.stack(channels)


def features(values, starts, length, horizon):
    # NumPy sliding windows and explicit edge replication avoid production's
    # torch avg_pool1d decomposition.
    windows = np.asarray(values[starts[:, None]+np.arange(length+horizon)], dtype=np.float64)
    x = windows[:, :length].transpose(0, 2, 1)
    padded = np.pad(x, ((0, 0), (0, 0), (12, 12)), mode="edge")
    smooth = np.lib.stride_tricks.sliding_window_view(padded, 25, axis=-1).mean(-1)
    design = np.empty((*x.shape[:2], 2*length+2), dtype=np.float64)
    design[:, :, :length] = x-smooth
    design[:, :, length] = 1
    design[:, :, length+1:-1] = smooth
    design[:, :, -1] = 1
    return design, windows[:, length:].transpose(0, 2, 1)


def b_autograd(values, starts, theta, length, horizon):
    parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    gradients = np.zeros((3, *theta.shape))
    losses = np.zeros(3)
    for lo in range(0, len(starts), 47):
        design, target = features(values, starts[lo:lo+47], length, horizon)
        error = torch.matmul(torch.from_numpy(design).permute(1, 0, 2), parameter).permute(1, 0, 2)-torch.from_numpy(target)
        for view, region in enumerate((slice(None), slice(horizon//2, None), slice(0, horizon//2))):
            loss = error[:, :, region].square().mean()
            gradient, = torch.autograd.grad(loss, parameter, retain_graph=view < 2)
            fraction = len(design)/len(starts)
            gradients[view] += gradient.detach().numpy()*fraction
            losses[view] += loss.item()*fraction
    norms = np.sqrt(np.sum(gradients*gradients, axis=(1, 2, 3)))
    direction = np.zeros_like(gradients)
    nonzero = norms > 0
    direction[nonzero] = gradients[nonzero]/norms[nonzero, None, None, None]
    return gradients, direction, norms, losses


def audit_parent(task, parent, data):
    a, b, ai, bi, first = data
    name = f"{task['task']}_s{parent['seed']}"
    record = read(HERE / (name+".json"))
    check("record_identity:"+name, record["task"] == task["task"] and record["family"] == task["family"]
          and record["seed"] == parent["seed"] and record["step"] == parent["step"])
    close("record_window_counts:"+name, [record["A_windows"], record["B_windows"]], [len(ai), len(bi)], rtol=0, atol=0)
    check("view_order:"+name, record["views"] == ["global", "target", "complement"])
    asset = ROOT / record["asset"]
    check("asset_hash:"+name, digest(asset) == record["sha256"] and asset.stat().st_size == record["bytes"])
    with np.load(asset, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    theta = load_theta(task, parent)
    close("theta:"+name, result["theta"], theta, rtol=0, atol=0)
    close("A_starts:"+name, result["A_starts"], ai, rtol=0, atol=0)
    close("B_starts:"+name, result["B_starts"], bi, rtol=0, atol=0)
    check("A_first:"+name, np.array_equal(first, result["A_first"]))
    length, horizon = task["length"], task["horizon"]
    bg, bd, bn, bl = b_autograd(b, bi, theta, length, horizon)
    close("B_gradient_autograd:"+name, result["B_gradient"], bg)
    close("B_direction_autograd:"+name, result["B_direction"], bd)
    close("B_norms:"+name, record["B_gradient_norms"], bn)
    close("B_losses:"+name, record["B_losses"], bl)
    close("B_reference_frozen_tolerance:"+name, bl[0], parent["reference_B"], rtol=1e-5, atol=1e-8)
    close("B_halves_identity:"+name, bg[0], (bg[1]+bg[2])/2)
    check("B_availability:"+name, record["direction_available"] == (bn > 0).tolist())
    n = len(ai)
    projections = np.zeros((n, 3))
    squared_norm = np.zeros(n)
    group_sum = np.zeros((2, *theta.shape))
    # Materialize each per-window parameter gradient. This avoids production's
    # factorized norm identity and projected-error contraction.
    for lo in range(0, n, 17):
        design, target = features(a, ai[lo:lo+17], length, horizon)
        pred = np.matmul(design.transpose(1, 0, 2), theta).transpose(1, 0, 2)
        gradients = design[:, :, :, None]*(pred-target)[:, :, None, :]*(2/(theta.shape[0]*horizon))
        flat = gradients.reshape(len(design), -1)
        projections[lo:lo+len(design)] = flat @ bd.reshape(3, -1).T
        squared_norm[lo:lo+len(design)] = np.square(flat).sum(axis=1)
        flags = first[lo:lo+len(design)]
        for group, mask in enumerate((flags, ~flags)):
            group_sum[group] += gradients[mask].sum(axis=0)
    rms = float(np.sqrt(squared_norm.mean()))
    close("A_gradient_rms:"+name, record["A_gradient_rms"], rms)
    check("A_availability:"+name, record["A_scale_available"] == (rms > 0))
    projection = projections/rms if rms else np.zeros_like(projections)
    normalized_norm = squared_norm/rms**2 if rms else np.zeros_like(squared_norm)
    close("all_A_projections:"+name, result["projection"], projection)
    close("all_A_gradient_norms:"+name, result["normalized_gradient_norm_square"], normalized_norm)
    counts = np.array([first.sum(), (~first).sum()])
    group_grad = group_sum/counts[:, None, None, None]
    group_grad = group_grad/rms if rms else np.zeros_like(group_grad)
    close("full_group_gradients:"+name, result["normalized_group_gradient"], group_grad)
    masks = (first, ~first)
    close("group_counts:"+name, record["group_counts"], counts, rtol=0, atol=0)
    close("group_mean:"+name, record["group_mean"], [projection[m].mean(0) for m in masks])
    close("group_second:"+name, record["group_second"], [(projection[m, :, None]*projection[m, None, :]).mean(0) for m in masks])
    close("group_norm_square_mean:"+name, record["group_norm_square_mean"], [normalized_norm[m].mean() for m in masks])
    sampled = sorted({0, n//2, n-1})
    for index in sampled:
        # Independent torch model and autograd, without the analytical feature
        # builder used for the full-array audit above.
        x = torch.tensor(a[ai[index]:ai[index]+length].T, dtype=torch.float64)
        extended = torch.cat([x[:, :1].expand(-1, 12), x, x[:, -1:].expand(-1, 12)], dim=1)
        smooth = extended.unfold(-1, 25, 1).mean(-1)
        parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
        prediction = torch.bmm((x-smooth)[:, None, :], parameter[:, :length]).squeeze(1)+parameter[:, length]
        prediction = prediction+torch.bmm(smooth[:, None, :], parameter[:, length+1:-1]).squeeze(1)+parameter[:, -1]
        target = torch.tensor(a[ai[index]+length:ai[index]+length+horizon].T, dtype=torch.float64)
        gradient, = torch.autograd.grad((prediction-target).square().mean(), parameter)
        gradient = gradient.numpy()
        close(f"A_autograd_projection:{name}:{index}", projection[index], (bd*gradient).sum((1, 2, 3))/rms if rms else np.zeros(3))
        close(f"A_autograd_norm:{name}:{index}", normalized_norm[index], np.square(gradient).sum()/rms**2 if rms else 0.)
    probabilities = {"PARENT": np.full(n, 1/n)}
    if task["family"] == "Hydraulic":
        for action, mass in [("KEEP_FIRST", [1, 0]), ("KEEP_SECOND", [0, 1]), ("FAVOR_FIRST", [.75, .25]), ("FAVOR_SECOND", [.25, .75])]:
            probabilities[action] = np.where(first, mass[0]/counts[0], mass[1]/counts[1])
    else:
        for action, weights in [("FAVOR_EARLY", [3, 1]), ("FAVOR_LATE", [1, 3])]:
            weight = np.where(first, weights[0], weights[1]).astype(np.float64)
            probabilities[action] = weight/weight.sum()
    check("action_row_count:"+name, len(record["actions"]) == 2*len(probabilities))
    for action, p in probabilities.items():
        close(f"probability_sum:{name}:{action}", p.sum(), 1.)
        mean = p @ projection
        masses = np.array([p[m].sum() for m in masks])
        mean_grad = (group_grad*(masses[:, None, None, None])).sum(0)
        mean_grad_norm = np.square(mean_grad).sum()
        for implementation in ("S", "W"):
            row = next(r for r in record["actions"] if r["action"] == action and r["implementation"] == implementation)
            q = p if implementation == "S" else n*p*p
            moment = (projection.T*q) @ projection
            covariance = moment-np.outer(mean, mean)
            full_second = q @ normalized_norm
            trace = full_second-mean_grad_norm
            prefix = f"{name}:{action}:{implementation}:"
            for field, expected in [("group_masses", masses), ("mean", mean), ("second", moment), ("covariance", covariance),
                                    ("batch256_covariance", covariance/256), ("normalized_gradient_second_moment", full_second),
                                    ("normalized_mean_gradient_norm_square", mean_grad_norm), ("normalized_gradient_covariance_trace", trace),
                                    ("normalized_batch256_trace", trace/256)]:
                close(prefix+field, row[field], expected)
            check(prefix+"PSD", np.linalg.eigvalsh(covariance).min() >= -1e-10*(1+np.abs(moment).max()))
            check(prefix+"trace_nonnegative", trace >= -1e-10*(1+full_second))
        s = next(r for r in record["actions"] if r["action"] == action and r["implementation"] == "S")
        w = next(r for r in record["actions"] if r["action"] == action and r["implementation"] == "W")
        close(f"S_W_mean_identity:{name}:{action}", s["mean"], w["mean"], rtol=0, atol=0)
        if action == "PARENT":
            close("PARENT_covariance_identity:"+name, s["covariance"], w["covariance"])
    return dict(parent=name, A_windows=n, B_windows=len(bi), autograd_A_indices=sampled,
                checked_action_rows=len(record["actions"]))


def full(lock):
    summary = read(HERE / "measurements.json")
    check("measurement_completed", summary["status"] == "completed" and len(summary["records"]) == 15)
    receipt = read(HERE / "source_publication_receipt.json")
    check("publication_gate", receipt["status"] == "exact_commit_and_tree_verified")
    records = []
    for task in lock["tasks"]:
        data = load_ab(task)
        for parent in task["parents"]:
            try:
                name = f"{task['task']}_s{parent['seed']}"
                expected_record = next(r for r in summary["records"] if r["task"] == task["task"] and r["seed"] == parent["seed"])
                check("summary_record_identity:"+name, read(HERE / (name+".json")) == expected_record)
                record = audit_parent(task, parent, data)
            except Exception:
                record = dict(parent=f"{task['task']}_s{parent['seed']}", error=traceback.format_exc())
            records.append(record)
            print(json.dumps(record), flush=True)
    for name, expected in {**lock["inputs"], **lock["scripts"]}.items():
        check("post_audit_sha256:"+name, digest(ROOT / name) == expected)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("gate", "full"))
    args = parser.parse_args()
    output = HERE / ("gate_audit.json" if args.action == "gate" else "audit.json")
    started = time.perf_counter()
    result = dict(status="failed", action=args.action, checks=CHECKS,
                  production_helpers_imported=False, C_D_test_numeric_members_parsed=False,
                  training_LLM_server_operations=0, source_audit_sha256=digest(Path(__file__)))
    try:
        lock = gate()
        if args.action == "full":
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            import numpy as np
            import torch
            torch.set_num_threads(1)
            result["parents"] = full(lock)
            result["environment"] = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__)
        result["status"] = "completed" if all(c["passed"] for c in CHECKS) and all("error" not in p for p in result.get("parents", [])) else "failed"
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["seconds"] = time.perf_counter()-started
        result["checks_passed"] = sum(c["passed"] for c in CHECKS)
        result["checks_total"] = len(CHECKS)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("status", "action", "seconds", "checks_passed", "checks_total")}), flush=True)
    if result["status"] != "completed":
        sys.exit(1)
