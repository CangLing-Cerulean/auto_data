"""Retrospective decision headroom from the complete, already disclosed S/W pool.

This does not access data, checkpoints, unrevealed labels, or remote services.
It leaves the original protocol, scores, and recommendation receipts unchanged.
"""

import csv
import hashlib
import json
import math
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research_runs/agent_research/intervention_diagnosis/20261014_weighted_response_r1"
OUTPUT = ROOT / "research_runs/agent_research/intervention_diagnosis/20261014_noise_necessity_review_r1"
TASKS = ("G1", "G2", "G3")
ACTIONS = ("KEEP_FIRST", "KEEP_SECOND", "FAVOR_FIRST", "FAVOR_SECOND")
SEEDS = (2026101701, 2026101702, 2026101703)
ADOPTION_PERCENT = 1.0  # Same M1 adoption rule as the existing evaluation.
MATERIAL_PP = 0.25  # Descriptive inherited band, not a new significance test.


def mean(values):
    return math.fsum(values) / len(values)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(rows):
    indexed = {(r["task"], int(r["seed"]), r["action"]): r for r in rows}
    expected = {(t, s, a) for t in TASKS for s in SEEDS for a in ACTIONS}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("Expected all 36 distinct task/seed/action pairs")
    for row in rows:
        for key in ("S_gain", "W_gain", "M1_gain", "gap_pp"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite {key}")
        if not math.isclose(float(row["W_gain"]) - float(row["S_gain"]),
                            float(row["gap_pp"]), abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("Stored paired gap differs from gains")

    records, cells = [], []
    for task in TASKS:
        gains, predictions = {}, {"PARENT": 0.0}
        for action in ACTIONS:
            group = [indexed[(task, seed, action)] for seed in SEEDS]
            predictions[action] = mean([float(r["M1_gain"]) for r in group])
            for implementation in ("S", "W"):
                values = [float(r[f"{implementation}_gain"]) for r in group]
                gains[(implementation, action)] = mean(values)
                cells.append({"task": task, "implementation": implementation,
                              "action": action, "mean_gain_percent": mean(values),
                              "M1_mean_prediction_percent": predictions[action],
                              "seed_gains_percent": values})
        # Order is inherited action order; no response-dependent tie breaking.
        chosen = max(ACTIONS, key=lambda a: predictions[a])
        if predictions[chosen] < ADOPTION_PERCENT:
            chosen = "PARENT"
        # The ex-post oracle includes PARENT even if every action is harmful.
        joint_candidates = [("PARENT", "PARENT", 0.0)] + [
            (implementation, action, gains[(implementation, action)])
            for implementation in ("S", "W") for action in ACTIONS
        ]
        oracle_implementation, oracle_action, oracle_gain = max(joint_candidates, key=lambda x: x[2])
        realized = {implementation: (0.0 if chosen == "PARENT" else gains[(implementation, chosen)])
                    for implementation in ("S", "W")}
        separate_oracle = {implementation: max([0.0] + [gains[(implementation, a)] for a in ACTIONS])
                           for implementation in ("S", "W")}
        gaps = ([0.0] * len(SEEDS) if chosen == "PARENT" else [
            float(indexed[(task, seed, chosen)]["W_gain"]) - float(indexed[(task, seed, chosen)]["S_gain"])
            for seed in SEEDS])
        sorted_predictions = sorted(predictions.values(), reverse=True)
        records.append({
            "task": task, "M1_action": chosen,
            "M1_selected_prediction_percent": predictions[chosen],
            "prediction_top_two_gap_pp": sorted_predictions[0] - sorted_predictions[1],
            "joint_oracle_implementation": oracle_implementation, "joint_oracle_action": oracle_action,
            "joint_oracle_gain_percent": oracle_gain,
            "S_priority_gain_percent": realized["S"], "W_priority_gain_percent": realized["W"],
            "S_priority_joint_regret_pp": oracle_gain - realized["S"],
            "W_priority_joint_regret_pp": oracle_gain - realized["W"],
            "S_separate_regret_pp": separate_oracle["S"] - realized["S"],
            "W_separate_regret_pp": separate_oracle["W"] - realized["W"],
            "selected_action_W_minus_S_mean_pp": mean(gaps),
            "selected_action_W_minus_S_seed_pp": gaps,
            "all_three_W_minus_S_above_material_band": all(g > MATERIAL_PP for g in gaps),
            "all_three_W_minus_S_below_negative_band": all(g < -MATERIAL_PP for g in gaps),
            "scope": "retrospective_three_seed_mean_decision; oracle_is_not_a_method"
        })
    aggregate = {key: mean([r[key] for r in records]) for key in (
        "joint_oracle_gain_percent", "S_priority_gain_percent", "W_priority_gain_percent",
        "S_priority_joint_regret_pp", "W_priority_joint_regret_pp")}
    aggregate["max_task_W_priority_joint_regret_pp"] = max(r["W_priority_joint_regret_pp"] for r in records)
    aggregate["W_priority_minus_S_priority_gain_pp"] = aggregate["W_priority_gain_percent"] - aggregate["S_priority_gain_percent"]
    aggregate["all_separate_regrets_zero"] = all(abs(r[f"{i}_separate_regret_pp"]) < 1e-12
                                                for r in records for i in ("S", "W"))
    return {"tasks": records, "cells": cells, "aggregate": aggregate}


def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = ("paired_seed_records.csv", "final_audit.json", "predictions.json", "PROTOCOL.md", "sync_receipt.json")
    before = {name: digest(SOURCE / name) for name in names}
    audit = json.loads((SOURCE / "final_audit.json").read_text(encoding="utf-8-sig"))
    if audit["status"] != "PASS":
        raise ValueError("Only audited disclosed labels may be used")
    with (SOURCE / "paired_seed_records.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = calculate(rows)
    after = {name: digest(SOURCE / name) for name in names}
    if before != after:
        raise ValueError("Source changed during analysis")
    result.update({"status": "completed_retrospective", "source_inputs_sha256": before,
                   "source_stage": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
                   "label_visibility": "all_labels_disclosed_before_this_analysis",
                   "new_full_trainings": 0, "cumulative_full_trainings": 409,
                   "test_access": False, "D_access": False,
                   "scientific_claim": "Known-pool opportunity ceiling only; no prospective validation, calibrated uncertainty or new method",
                   "analysis_seconds": time.perf_counter() - started})
    with (OUTPUT / "decision_value.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (OUTPUT / "joint_decisions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[k for k, v in result["tasks"][0].items() if not isinstance(v, list)])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if not isinstance(v, list)} for row in result["tasks"])
    print(json.dumps(result["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
