"""Read-only-input audit and exact certificate for the failed SLSQP status check."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    root = Path(__file__).resolve().parents[1]
    output = stage / "artifact_audit.json"
    if output.exists():
        raise FileExistsError(output)
    lock = json.loads((stage / "source_lock_r2.json").read_text(encoding="utf-8-sig"))
    checks = []
    for entry in lock["files"]:
        digest = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
        checks.append({"name": entry["path"], "pass": digest == entry["sha256"]})
    result_path = stage / "algebra_check_r2.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    failed = [x for x in result["checks"] if not x["pass"]]
    checks.append({"name": "original_failure_preserved", "pass": result["status"] == "FAIL"
        and len(result["checks"]) == 112 and [x["name"] for x in failed] == ["signed_solver_2"]})
    row = failed[0]
    # a=(-1,2,0): d(a1/p1)/dp1>0; d(a2/p2)/dp2<0.
    # Moving any surplus of p1 or p3 to p2 strictly lowers the objective.
    # Thus p1=l1, p3=l3 is globally optimal, not merely stationary.
    probability = [Fraction(1, 10), Fraction(13, 20), Fraction(1, 4)]
    coefficients = [Fraction(-1), Fraction(2), Fraction(0)]
    value = sum(a/p for a, p in zip(coefficients, probability))
    checks.append({"name": "exact_global_certificate_for_signed_solver_2",
        "pass": row["a"] == [-1, 2, 0]
            and row["P"] == list(map(float, probability))
            and abs(row["objective"] - float(value)) <= 1e-12
            and abs(row["grid_best"] - float(value)) <= 1e-12
            and all(abs(x-float(value)) <= 1e-8 for x in row["scipy_values"]),
        "exact_P": [str(p) for p in probability], "exact_objective": str(value),
        "proof": "Any surplus above p1 or p3 floors can be moved to positive-coefficient p2, strictly decreasing sum(a/p); therefore p=(1/10,13/20,1/4) is the global minimizer.",
        "original_SLSQP_statuses_unchanged": row["scipy_success"],
        "scope": "new retrospective certificate; does not change the original 111/112 or mark its check PASS"})
    rows = result["exact_enumeration_descriptive_only"]
    expected = {(seed, policy) for seed in (10301, 10302, 10303)
        for policy in ("uniform", "gradient_norm", "jacobian_norm", "geometry", "full")}
    checks.append({"name": "all_enumeration_rows_retained", "pass": len(rows) == 15
        and {(x["seed"], x["policy"]) for x in rows} == expected})
    source1 = json.loads((stage / "source_lock.json").read_text(encoding="utf-8-sig"))
    first_checker = next(x["sha256"] for x in source1["files"]
        if x["path"] == "scripts/check_adam_risk_sampling.py")
    checks.append({"name": "attempt1_checker_preserved", "pass": hashlib.sha256(
        (stage / "failed_check_v1.py.txt").read_bytes()).hexdigest() == first_checker})
    artifact = {"status": "PASS" if all(x["pass"] for x in checks) else "FAIL",
        "original_check_status": "FAIL", "original_check_passed": 111,
        "original_check_total": 112, "scope": "asset integrity and one exact mathematical certificate, not independent rerun",
        "input_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "checks": checks, "real_diagnostic_enabled": False,
        "scientific_advantage_established": False}
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "original_check_status": "FAIL",
                      "certificate": str(value)}))
    return int(artifact["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
