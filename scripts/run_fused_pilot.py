"""Fixed-pool, no-API pilot. Execute only on the designated server.

Uses the immutable original DLinear class/evaluator, never its main runner.
Each invocation prepares data or trains exactly one new registered run.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def check_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Protected identity mismatch: {path}")


def original_core(p):
    base = Path(p["base_root"])
    check_hash(base / "research_runs/protocol.json", p["base_protocol_sha256"])
    code = base / "scripts/run_research.py"
    check_hash(code, p["core_sha256"])
    spec = importlib.util.spec_from_file_location("frozen_original_core", code)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def recent_switch(raw, starts):
    changed = np.any(np.diff(raw[:, 7:], axis=0) != 0, axis=1)
    cumulative = np.r_[0, np.cumsum(changed)]
    return cumulative[starts + 31] > cumulative[starts + 27]


def variation(values, starts):
    cumulative = np.r_[0.0, np.cumsum(np.abs(np.diff(values.astype(np.float64), axis=0)).mean(axis=1))]
    return (cumulative[starts + 31] - cumulative[starts]) / 31


def overlap(left, right):
    both = int(np.sum(left & right))
    union = int(np.sum(left | right))
    return {"left": int(left.sum()), "right": int(right.sum()), "intersection": both,
            "jaccard": both / union if union else None,
            "left_covered": both / int(left.sum()) if left.any() else None,
            "right_covered": both / int(right.sum()) if right.any() else None}


def weights(mask, factor):
    raw = np.where(mask, factor, 1.0)
    return raw / raw.mean()


def prepare(protocol_path):
    begun = time.monotonic()
    p = load(protocol_path)
    if p["deepseek_api_enabled"] or p["api_provider_calls_allowed"] or p["test_access"]:
        raise ValueError("Pilot boundary violated")
    root, base = Path(p["output_root"]), Path(p["base_root"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "preflight.json").exists():
        raise ValueError("Preflight already exists; inspect its receipt instead of overwriting")
    core = original_core(p)
    manifest_path = base / "data/metropt3_v1/manifest.json"
    check_hash(manifest_path, p["manifest_sha256"])
    manifest = load(manifest_path)
    verified = {}
    for name in ("train", "validation"):
        entry = next(s for s in manifest["splits"] if s["name"] == name)
        file = base / "data/metropt3_v1" / entry["file"]
        check_hash(file, entry["sha256"])
        verified[name] = {"sha256": entry["sha256"], "bytes": file.stat().st_size}
    train_path = base / "data/metropt3_v1/train.csv"
    raw = np.loadtxt(train_path, delimiter=",", skiprows=1, usecols=range(2, 17), dtype=np.float64)
    with train_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [(int(row[0]), row[1]) for row in reader]
    ids = np.array([row[0] for row in rows], dtype=np.int64)
    times = np.array([row[1] for row in rows], dtype="datetime64[s]")
    expected = next(s for s in manifest["splits"] if s["name"] == "train")
    if (header != manifest["columns_raw"] or len(raw) != expected["rows"]
            or not np.isfinite(raw).all() or not np.all(np.diff(ids) > 0)):
        raise ValueError("Raw training data contract mismatch")
    if (str(times[0]).replace("T", " ") != expected["first_timestamp"]
            or str(times[-1]).replace("T", " ") != expected["last_timestamp"]):
        raise ValueError("Training timestamps mismatch")
    if manifest["sensor_columns"][7:] != p["selector_channels"]:
        raise ValueError("Selector channel mapping mismatch")
    all_starts = core.valid_starts(times, 40, 30)
    summary = {"status": "passed", "scope": "manifest_core_train_validation_hashes_and_train_rows_only",
               "full_three_partition_verification": False, "test_opened": False,
               "protocol_sha256": sha(protocol_path), "runner_sha256": sha(__file__),
               "verified_files": verified, "tasks": {}, "new_trainings": 0, "API_calls": 0}
    for task_id, task in p["tasks"].items():
        cut, end = task["train_row_range"][1], task["development_row_range"][1]
        if task["train_row_range"][0] != 0 or task["development_row_range"][0] != cut:
            raise ValueError("Unexpected task range")
        starts = all_starts[all_starts + 40 <= cut]
        dev_starts = all_starts[(all_starts >= cut) & (all_starts + 40 <= end)]
        if not len(starts) or not len(dev_starts) or starts[-1] + 40 > dev_starts[0]:
            raise ValueError("Invalid or overlapping task windows")
        mean, std = raw[:cut].mean(0), raw[:cut].std(0)
        scale = np.where(std == 0, 1, std)
        train = ((raw[:cut] - mean) / scale).astype(np.float32)
        dev = ((raw[:end] - mean) / scale).astype(np.float32)
        target, dev_target = recent_switch(raw, starts), recent_switch(raw, dev_starts)
        if not target.any() or target.all() or not dev_target.any() or dev_target.all():
            raise ValueError("Selector lacks both target and complement coverage")
        months = times[starts + 32].astype("datetime64[M]").astype(str)
        comp = raw[starts + 31, 7]
        if not np.isin(comp, [0., 1.]).all():
            raise ValueError("COMP must have registered binary strata")
        strata = np.char.add(np.char.add(months, ":"), comp.astype(np.int64).astype(str))
        rng = np.random.default_rng(p["control_seed"])
        matched = np.zeros(len(starts), bool)
        strata_audit = {}
        for stratum in np.unique(strata):
            members = np.flatnonzero(strata == stratum)
            selected = rng.permutation(members)[:int(target[members].sum())]
            matched[selected] = True
            strata_audit[str(stratum)] = {"windows": len(members), "target": int(target[members].sum()), "control": len(selected)}
        if int(matched.sum()) != int(target.sum()) or np.array_equal(matched, target):
            raise ValueError("Non-targeted control is degenerate")
        tv = variation(train, starts)
        low, high = np.quantile(tv, [.25, .9])
        history = {"current_T_low_variation_proxy": overlap(target, tv <= low),
                   "current_T_high_variation_proxy": overlap(target, tv > high)}
        for mask_path in sorted((base / "research_runs/cluster_selection").glob("mask_*_target.npz")):
            receipt = load(mask_path.with_suffix(".json"))
            check_hash(mask_path, receipt["asset_sha256"])
            if receipt["manifest_sha256"] != p["manifest_sha256"]:
                raise ValueError("Historical mask dataset mismatch")
            with np.load(mask_path, allow_pickle=False) as old:
                old_starts, removed = old["start_row"], ~old["retained"]
                common, new_idx, old_idx = np.intersect1d(starts, old_starts, return_indices=True)
                if len(common):
                    history[mask_path.name] = {"asset_sha256": receipt["asset_sha256"],
                        "comparison_scope": "shared_legal_source_start_rows", "common_windows": len(common),
                        "overlap": overlap(target[new_idx], removed[old_idx]),
                        "used_to_define_new_weights": False}
        action_audit = {}
        for action, factor in p["actions"].items():
            mask = matched if action.startswith("CTRL_") else target
            w = weights(mask, factor)
            if not np.isfinite(w).all() or w.min() <= 0 or abs(w.mean() - 1) > 1e-12:
                raise ValueError("Weight normalization contract failure")
            action_audit[action] = {"weight_min": float(w.min()), "weight_max": float(w.max()),
                "weight_mean": float(w.mean()), "ESS": float(w.sum() ** 2 / np.square(w).sum()),
                "retained_windows": len(w), "weights_sha256": array_sha(w)}
        folder = root / "tasks" / task_id
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / "prepared.npz", train=train, development=dev, train_starts=starts,
            development_starts=dev_starts, target=target, control=matched, development_target=dev_target,
            source_ids=ids[:end], timestamps=times[:end].astype(np.int64), mean=mean, scale=scale,
            train_months=months)
        audit = {"task_id": task_id, "source_group_id": "metropt3", "audit_interval": None,
            "train_row_range": [0, cut], "development_row_range": [cut, end],
            "training_time_range": [str(times[0]), str(times[cut-1])],
            "development_time_range": [str(times[cut]), str(times[end-1])],
            "train_windows": len(starts), "development_windows": len(dev_starts),
            "train_target_fraction": float(target.mean()), "development_target_fraction": float(dev_target.mean()),
            "mean": mean.tolist(), "scale": scale.tolist(), "strata": strata_audit,
            "target_control_overlap": overlap(target, matched), "historical_overlap": history,
            "historical_overlap_scope": "available_cluster_target_masks_and_current_T_variation_proxies_not_exhaustive_history",
            "action_audit": action_audit, "prepared_sha256": sha(folder / "prepared.npz"),
            "new_sensor_records": 0, "all_original_evaluation_windows_preserved": True}
        write(folder / "audit.json", audit)
        summary["tasks"][task_id] = audit
    for name, identity in verified.items():
        check_hash(base / "data/metropt3_v1" / f"{name}.csv", identity["sha256"])
    summary["elapsed_seconds"] = time.monotonic() - begun
    write(root / "preflight.json", summary)
    print(json.dumps({"status": "preflight_passed", "tasks": {k: {f: v[f] for f in ("train_windows", "development_windows", "train_target_fraction", "development_target_fraction")} for k, v in summary["tasks"].items()}, "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


def train(protocol_path, task_id, action, seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    p = load(protocol_path)
    root, base = Path(p["output_root"]), Path(p["base_root"])
    preflight = load(root / "preflight.json")
    if preflight["protocol_sha256"] != sha(protocol_path) or preflight["runner_sha256"] != sha(__file__):
        raise ValueError("Preflight identity changed")
    if p["deepseek_api_enabled"] or p["api_provider_calls_allowed"] or p["test_access"]:
        raise ValueError("Pilot boundary violated")
    if task_id not in p["tasks"] or action not in p["actions"] or seed not in [p["exploration_seed"], *p["confirmation_seeds"]]:
        raise ValueError("Unregistered task/action/seed")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
        raise RuntimeError("Designated 4090 required")
    run_id = f"FPR1_{task_id}_{action}_s{seed}"
    folder = root / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    write(folder / "identity.json", {"run_id": run_id, "task_id": task_id, "action": action, "seed": seed,
        "protocol_sha256": sha(protocol_path), "runner_sha256": sha(__file__), "API_calls": 0})
    (folder / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (folder / "protocol.json").write_bytes(Path(protocol_path).read_bytes())
    last_step = 0
    try:
        core = original_core(p)
        check_hash(base / "data/metropt3_v1/manifest.json", p["manifest_sha256"])
        for name, identity in preflight["verified_files"].items():
            check_hash(base / "data/metropt3_v1" / f"{name}.csv", identity["sha256"])
        task_folder = root / "tasks" / task_id
        audit = load(task_folder / "audit.json")
        check_hash(task_folder / "prepared.npz", audit["prepared_sha256"])
        with np.load(task_folder / "prepared.npz", allow_pickle=False) as packed:
            train_data = torch.from_numpy(packed["train"]).cuda()
            development_data = torch.from_numpy(packed["development"]).cuda()
            starts_np = packed["train_starts"]
            development_starts_np = packed["development_starts"]
            starts = torch.from_numpy(starts_np).cuda()
            development_starts = torch.from_numpy(development_starts_np).cuda()
            target, control, dev_target = packed["target"], packed["control"], packed["development_target"]
            source_ids, timestamps = packed["source_ids"], packed["timestamps"].astype("datetime64[s]")
        w = weights(control if action.startswith("CTRL_") else target, p["actions"][action])
        if array_sha(w) != audit["action_audit"][action]["weights_sha256"]:
            raise ValueError("Instantiated weight identity changed")
        train_weights = torch.from_numpy(w.astype(np.float32)).cuda()
        groups = {"all": np.ones(len(development_starts_np), bool), "recent_switch": dev_target, "not_recent_switch": ~dev_target}
        months = timestamps[development_starts_np + 32].astype("datetime64[M]").astype(str)
        groups.update({"month_" + m: months == m for m in np.unique(months)})
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        f = p["fixed_training"]
        model = core.DLinear(f).cuda()
        if sum(t.numel() for t in model.parameters()) != f["parameters"]:
            raise ValueError("Model parameter count changed")
        torch.save(model.state_dict(), folder / "initial.pt")
        initial_hash = hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
        optimizer = torch.optim.Adam(model.parameters(), lr=f["learning_rate"], weight_decay=f["weight_decay"])
        generator = torch.Generator(device="cuda").manual_seed(seed + 1000)
        counts = torch.zeros(len(starts_np), dtype=torch.int64, device="cuda")
        offsets = torch.arange(40, device="cuda")
        history, best, best_step = [], float("inf"), None
        write(folder / "status.json", {"status": "training", "step": 0})
        for step in range(1, f["steps"] + 1):
            sample = torch.randint(len(starts_np), (f["batch_size"],), generator=generator, device="cuda")
            counts.scatter_add_(0, sample, torch.ones_like(sample))
            batch = train_data[starts[sample, None] + offsets]
            optimizer.zero_grad(set_to_none=True)
            error = (model(batch[:, :32]) - batch[:, 32:]).square().mean(dim=(1, 2))
            loss = (error * train_weights[sample]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            last_step = step
            if step % f["validation_every"] == 0:
                metrics = core.evaluate(model, development_data, development_starts, groups, f, "baseline")
                history.append({"step": step, "weighted_training_batch_loss": float(loss.detach()), "metrics": metrics})
                score = metrics["groups"]["all"]["NMSE"]
                if score < best:
                    best, best_step = score, step
                    torch.save(model.state_dict(), folder / "best.pt")
                write(folder / "history.json", history)
                write(folder / "status.json", {"status": "training", "step": step, "elapsed_seconds": time.monotonic() - begun})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampling_rng": generator.get_state(),
                    "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, folder / "final_training_state.pt")
        torch.save(model.state_dict(), folder / "final.pt")
        final_metrics = history[-1]["metrics"]
        model.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
        metrics = core.evaluate(model, development_data, development_starts, groups, f, "baseline")
        persistence = core.evaluate(model, development_data, development_starts, groups, f, "baseline", persistence=True)
        draws = counts.cpu().numpy()
        if int(draws.sum()) != 768000:
            raise ValueError("Sample exposure changed")
        np.savez_compressed(folder / "sampling.npz", start_row=starts_np, source_start=source_ids[starts_np],
                            draw_count=draws, weights=w, target_mask=target, control_mask=control)
        np.savez_compressed(folder / "evaluation_mapping.npz", start_row=development_starts_np,
                            source_start=source_ids[development_starts_np], target_start=source_ids[development_starts_np + 32],
                            source_end=source_ids[development_starts_np + 39], recent_switch=dev_target)
        for name, identity in preflight["verified_files"].items():
            check_hash(base / "data/metropt3_v1" / f"{name}.csv", identity["sha256"])
        check_hash(base / "scripts/run_research.py", p["core_sha256"])
        partition = "confirmation_development" if seed in p["confirmation_seeds"] else "development"
        result = {"run_id": run_id, "task_id": task_id, "action": action, "training_repeat": seed,
            "baseline_run_id": f"FPR1_{task_id}_B0_s{seed}", "evaluation_partition": partition,
            "execution_status": "succeeded", "protocol_validity": "valid", "scientific_status": "unassessed",
            "guardrail_status": "not_assessed", "visible_to_policy_at": {}, "cost_record_id": run_id + ":cost",
            "protocol_sha256": sha(protocol_path), "data_audit_sha256": sha(task_folder / "audit.json"),
            "metrics": metrics, "final_step_metrics": final_metrics, "persistence": persistence,
            "best_step": best_step, "steps": 3000, "samples_presented": int(draws.sum()),
            "draw_count_sha256": array_sha(draws), "initial_model_sha256": initial_hash,
            "elapsed_seconds": time.monotonic() - begun, "computed_at": datetime.now(timezone.utc).isoformat(), "DeepSeek_API_calls": 0,
            "environment": {"torch": torch.__version__, "numpy": np.__version__, "GPU": torch.cuda.get_device_name()},
            "assets": {x.name: sha(x) for x in folder.iterdir() if x.is_file() and x.name != "status.json"}}
        write(folder / "result.json", result)
        write(folder / "status.json", {"status": "completed", "step": last_step, "elapsed_seconds": result["elapsed_seconds"]})
        print(json.dumps({k: result[k] for k in ("run_id", "metrics", "elapsed_seconds", "best_step")}), flush=True)
    except BaseException as exc:
        write(folder / "failure.json", {"type": type(exc).__name__, "message": str(exc), "step": last_step,
            "elapsed_seconds": time.monotonic() - begun, "scientific_status": "not_assessable", "API_calls": 0})
        write(folder / "status.json", {"status": "failed", "step": last_step})
        raise


def timeout_handler(signum, frame):
    raise TimeoutError("Registered single-job time limit reached")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--task", choices=("O60", "O80"))
    parser.add_argument("--action", choices=("B0", "UP", "DOWN", "CTRL_UP", "CTRL_DOWN"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    protocol = load(args.protocol)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(protocol["per_preparation_seconds"] if args.prepare else protocol["per_training_seconds"])
    if args.prepare:
        prepare(args.protocol)
    else:
        if args.task is None or args.action is None or args.seed is None:
            parser.error("Training requires task, action and seed")
        train(args.protocol, args.task, args.action, args.seed)
