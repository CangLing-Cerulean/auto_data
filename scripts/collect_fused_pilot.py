"""Read-only analysis of pilot run assets; write derived receipts, never run models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(result, scope="all"):
    return result["metrics"]["groups"][scope]


def compare(base, candidate):
    return {"NMSE_gain_pct": 100 * (1 - metrics(candidate)["NMSE"] / metrics(base)["NMSE"]),
            "NMAE_deterioration_pct": 100 * (metrics(candidate)["NMAE"] / metrics(base)["NMAE"] - 1),
            "local_NMSE_gain_pct": 100 * (1 - metrics(candidate, "recent_switch")["NMSE"] / metrics(base, "recent_switch")["NMSE"]),
            "other_NMSE_gain_pct": 100 * (1 - metrics(candidate, "not_recent_switch")["NMSE"] / metrics(base, "not_recent_switch")["NMSE"])}


def main(root):
    begun = time.monotonic()
    p, preflight = load(root / "protocol.json"), load(root / "preflight.json")
    records, failures, costs, by_key = [], [], [], {}
    for folder in sorted((root / "runs").glob("FPR1_*")):
        path = folder / "result.json"
        if not path.exists():
            failure_path = folder / "failure.json"
            if failure_path.exists():
                failure = load(failure_path)
                failures.append({"run_id": folder.name, **failure})
                costs.append({"cost_record_id": folder.name + ":cost", "category": "failed_training_process",
                              "elapsed_seconds": failure["elapsed_seconds"], "DeepSeek_API_calls": 0})
            else:
                failures.append({"run_id": folder.name, "status": "incomplete_unassessed"})
            continue
        r = load(path)
        if r["protocol_sha256"] != sha(root / "protocol.json"):
            raise ValueError("Protocol identity mismatch")
        for name, expected in r["assets"].items():
            if sha(folder / name) != expected:
                raise ValueError(f"Corrupted run artifact: {folder.name}/{name}")
        records.append(r)
        by_key[(r["task_id"], r["action"], r["training_repeat"])] = r
        costs.append({"cost_record_id": r["cost_record_id"], "category": "completed_training_process",
                      "elapsed_seconds": r["elapsed_seconds"], "DeepSeek_API_calls": 0,
                      "includes_loading_evaluation_and_asset_writes": True, "GPU_kernel_seconds": None})
    cost_total = sum(c["elapsed_seconds"] for c in costs)
    costs.append({"cost_record_id": "preflight", "category": "data_preparation",
                  "elapsed_seconds": preflight["elapsed_seconds"], "DeepSeek_API_calls": 0})
    exploration = {}
    eligible_effect = False
    for task in p["tasks"]:
        base = by_key.get((task, "B0", p["exploration_seed"]))
        if base is None:
            continue
        exploration[task] = {}
        for action in p["actions"]:
            r = by_key.get((task, action, p["exploration_seed"]))
            if r is None:
                continue
            if r["draw_count_sha256"] != base["draw_count_sha256"] or r["initial_model_sha256"] != base["initial_model_sha256"]:
                raise ValueError("Paired randomization mismatch")
            diff = compare(base, r)
            exploration[task][action] = {**diff, "best_step": r["best_step"], "raw_overall": metrics(r),
                                        "raw_local": metrics(r, "recent_switch")}
            if action in ("UP", "DOWN") and diff["NMSE_gain_pct"] >= 1 and diff["NMAE_deterioration_pct"] <= 1:
                eligible_effect = True
    gaps = {}
    for task in p["tasks"]:
        e = exploration.get(task, {})
        if "UP" in e and "DOWN" in e:
            gaps[task] = e["UP"]["NMSE_gain_pct"] - e["DOWN"]["NMSE_gain_pct"]
    reversal = (len(gaps) == 2 and gaps["O60"] * gaps["O80"] < 0
                and min(abs(gaps["O60"]), abs(gaps["O80"])) >= 1)
    confirmations = {}
    for task in p["tasks"]:
        confirmations[task] = {}
        for action in p["actions"]:
            pairs = [(by_key.get((task, "B0", seed)), by_key.get((task, action, seed))) for seed in p["confirmation_seeds"]]
            if not all(b is not None and r is not None for b, r in pairs):
                continue
            for b, r in pairs:
                if r["draw_count_sha256"] != b["draw_count_sha256"] or r["initial_model_sha256"] != b["initial_model_sha256"]:
                    raise ValueError("Confirmation pairing mismatch")
            avg = lambda which, scope, key: sum(metrics(pair[which], scope)[key] for pair in pairs) / len(pairs)
            gain = 100 * (1 - avg(1, "all", "NMSE") / avg(0, "all", "NMSE"))
            mae = 100 * (avg(1, "all", "NMAE") / avg(0, "all", "NMAE") - 1)
            local = 100 * (1 - avg(1, "recent_switch", "NMSE") / avg(0, "recent_switch", "NMSE"))
            per_seed = [compare(b, r) for b, r in pairs]
            confirmations[task][action] = {"mean_NMSE_gain_pct": gain, "mean_NMAE_deterioration_pct": mae,
                "local_NMSE_gain_pct": local, "per_seed": per_seed,
                "effective": gain >= 1 and mae <= 1 and all(d["NMSE_gain_pct"] > 0 for d in per_seed),
                "harmful": gain <= -1 and all(d["NMSE_gain_pct"] < 0 for d in per_seed),
                "guardrail_pass": mae <= 1, "not_a_population_significance_test": True}
    matrix_complete = all((task, action, p["exploration_seed"]) in by_key for task in p["tasks"] for action in p["actions"])
    summary = {"study_id": p["study_id"], "computed_at": datetime.now(timezone.utc).isoformat(),
        "completed_trainings": len(records), "failed_or_incomplete": failures,
        "training_process_seconds": cost_total, "preflight_seconds": preflight["elapsed_seconds"],
        "DeepSeek_API_calls": 0, "internal_agent_usage": "separate_unknown_billing_not_free",
        "exploration": exploration, "rank_gap_UP_minus_DOWN_pct": gaps,
        "confirmation_gate": {"exploration_matrix_complete": matrix_complete, "eligible_effect": eligible_effect, "rank_reversal": reversal,
                              "passes": (eligible_effect or reversal) if matrix_complete else None}, "confirmation": confirmations,
        "paired_initialization_and_exposure": "passed_for_available_pairs",
        "independent_data": False, "test_access": False,
        "collector_seconds": time.monotonic() - begun}
    write(root / "summary.json", summary)
    for name, rows in (("responses.jsonl", records), ("costs.jsonl", costs)):
        (root / name).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
