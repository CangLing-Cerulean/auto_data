"""Independent checks of a retrospective decision ceiling and algebraic example."""

import hashlib
import json
import math
import statistics
import time
from decimal import Decimal, localcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "research_runs/agent_research/intervention_diagnosis/20261014_noise_necessity_review_r1"
SOURCE = ROOT / "research_runs/agent_research/intervention_diagnosis/20261014_weighted_response_r1"


def check_close(left, right):
    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError((left, right))


def adam_example():
    """Finite binomial sums, not synthetic sensor records or model training."""
    batch = 256
    epsilon = Decimal("1e-8")
    outputs = []
    with localcontext() as ctx:
        ctx.prec = 60
        for name, p, lo, hi in (
            ("symmetric", Decimal("0.5"), Decimal(-1), Decimal(1)),
            ("asymmetric", Decimal("0.2"), Decimal("-0.5"), Decimal(2)),
        ):
            first = (1 - p) * lo + p * hi
            second = (1 - p) * lo * lo + p * hi * hi
            probabilities, terms = [], []
            for k in range(batch + 1):
                probability = Decimal(math.comb(batch, k)) * p**k * (1 - p)**(batch - k)
                g = lo + (hi - lo) * Decimal(k) / Decimal(batch)
                probabilities.append(probability)
                terms.append(probability * g / (abs(g) + epsilon))
            assert first == 0 and second == 1
            assert abs(sum(probabilities) - 1) < Decimal("1e-55")
            output = sum(terms)
            outputs.append({"distribution": name, "positive_probability": str(p),
                            "low": str(lo), "high": str(hi), "per_example_mean": str(first),
                            "per_example_variance": str(second), "batch_mean_variance": str(second / batch),
                            "expected_first_adam_normalized_gradient": str(output)})
        assert abs(Decimal(outputs[0]["expected_first_adam_normalized_gradient"])) < Decimal("1e-55")
        assert Decimal(outputs[1]["expected_first_adam_normalized_gradient"]) < Decimal("-0.049")
    return {"type": "algebraic_counterexample", "batch_size": batch,
            "adam_epsilon": str(epsilon), "bias_correction": True,
            "initial_optimizer_state": "zero; weight_decay=0",
            "formula": "first update = -learning_rate * gbar / (abs(gbar)+epsilon)",
            "distributions": outputs,
            "scope": "Same first two gradient moments need not imply same expected Adam update; no claim that these distributions occur in the research datasets"}


def main():
    started = time.perf_counter()
    source = json.loads((SOURCE / "final_audit.json").read_text(encoding="utf-8-sig"))
    result_path = STAGE / "decision_value.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert source["status"] == "PASS"
    assert len(source["records"]) == 36 and len(result["tasks"]) == 3 and len(result["cells"]) == 24
    actions = ("KEEP_FIRST", "KEEP_SECOND", "FAVOR_FIRST", "FAVOR_SECOND")
    tasks = ("G1", "G2", "G3")
    seeds = (2026101701, 2026101702, 2026101703)
    assert {(r["task"], r["seed"], r["action"]) for r in source["records"]} == {
        (task, seed, action) for task in tasks for seed in seeds for action in actions}
    assert {r["task"] for r in result["tasks"]} == set(tasks)
    assert {(r["task"], r["implementation"], r["action"]) for r in result["cells"]} == {
        (task, implementation, action) for task in tasks for implementation in ("S", "W") for action in actions}
    for name, digest in result["source_inputs_sha256"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
    recomputed = []
    for task in ("G1", "G2", "G3"):
        original = [r for r in source["records"] if r["task"] == task]
        assert len(original) == 12
        order = ("KEEP_FIRST", "KEEP_SECOND", "FAVOR_FIRST", "FAVOR_SECOND")
        scores = {a: statistics.mean(r["M1_gain"] for r in original if r["action"] == a) for a in order}
        chosen = max(order, key=scores.get)
        if scores[chosen] < 1:
            chosen = "PARENT"
        truth = {i: {a: statistics.mean(r[f"{i}_gain"] for r in original if r["action"] == a)
                     for a in order} for i in ("S", "W")}
        for i in truth:
            truth[i]["PARENT"] = 0
        oracle = max(value for mapping in truth.values() for value in mapping.values())
        stored = next(r for r in result["tasks"] if r["task"] == task)
        assert stored["M1_action"] == chosen
        check_close(stored["joint_oracle_gain_percent"], oracle)
        if stored["joint_oracle_implementation"] == "PARENT":
            assert stored["joint_oracle_action"] == "PARENT"
            check_close(oracle, 0.0)
        else:
            assert stored["joint_oracle_implementation"] in ("S", "W")
            assert stored["joint_oracle_action"] in order
            check_close(truth[stored["joint_oracle_implementation"]][stored["joint_oracle_action"]], oracle)
        check_close(stored["M1_selected_prediction_percent"], 0.0 if chosen == "PARENT" else scores[chosen])
        sorted_scores = sorted([0.0, *scores.values()], reverse=True)
        check_close(stored["prediction_top_two_gap_pp"], sorted_scores[0] - sorted_scores[1])
        selected_rows = {r["seed"]: r for r in original if r["action"] == chosen}
        gaps = [0.0] * 3 if chosen == "PARENT" else [selected_rows[s]["W_gain"] - selected_rows[s]["S_gain"] for s in seeds]
        assert len(stored["selected_action_W_minus_S_seed_pp"]) == 3
        for actual, expected in zip(stored["selected_action_W_minus_S_seed_pp"], gaps):
            check_close(actual, expected)
        check_close(stored["selected_action_W_minus_S_mean_pp"], statistics.mean(gaps))
        assert stored["all_three_W_minus_S_above_material_band"] == all(g > .25 for g in gaps)
        assert stored["all_three_W_minus_S_below_negative_band"] == all(g < -.25 for g in gaps)
        for implementation in ("S", "W"):
            gain = truth[implementation][chosen]
            check_close(stored[f"{implementation}_priority_gain_percent"], gain)
            check_close(stored[f"{implementation}_priority_joint_regret_pp"], oracle - gain)
            check_close(stored[f"{implementation}_separate_regret_pp"], max(truth[implementation].values()) - gain)
            assert oracle - gain >= -1e-12
        for cell in [r for r in result["cells"] if r["task"] == task]:
            check_close(cell["mean_gain_percent"], truth[cell["implementation"]][cell["action"]])
            check_close(cell["M1_mean_prediction_percent"], scores[cell["action"]])
            seed_values = {r["seed"]: r[f'{cell["implementation"]}_gain'] for r in original if r["action"] == cell["action"]}
            assert len(cell["seed_gains_percent"]) == 3
            for actual, seed in zip(cell["seed_gains_percent"], seeds):
                check_close(actual, seed_values[seed])
        recomputed.append(stored)
    for key in ("joint_oracle_gain_percent", "S_priority_gain_percent", "W_priority_gain_percent",
                "S_priority_joint_regret_pp", "W_priority_joint_regret_pp"):
        check_close(result["aggregate"][key], statistics.mean(row[key] for row in recomputed))
    check_close(result["aggregate"]["max_task_W_priority_joint_regret_pp"],
                max(row["W_priority_joint_regret_pp"] for row in recomputed))
    check_close(result["aggregate"]["W_priority_minus_S_priority_gain_pp"],
                result["aggregate"]["W_priority_gain_percent"] - result["aggregate"]["S_priority_gain_percent"])
    assert result["aggregate"]["all_separate_regrets_zero"] == all(
        abs(row[f"{implementation}_separate_regret_pp"]) < 1e-12 for row in recomputed for implementation in ("S", "W"))
    example = adam_example()
    receipt = {"status": "PASS", "source_records": 36, "joint_pool_cells": 24,
               "task_decisions": 3, "source_hashes_unchanged": True,
               "decision_value_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
               "counterexample": example, "audit_seconds": time.perf_counter() - started,
               "new_full_trainings": 0, "new_optimizer_updates": 0,
               "D_access": False, "test_access": False}
    with (STAGE / "audit.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "counterexample"}))


if __name__ == "__main__":
    main()
