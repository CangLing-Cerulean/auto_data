"""Generated-data checks only. Run from any directory with NumPy installed."""

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from coldstart_reference import conditional_z, evidence, predict, student_t_logpdf


def main():
    checks = []

    def equal(name, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
        checks.append(name)

    def rejects(name, function):
        try:
            function()
        except (ValueError, TypeError, KeyError):
            checks.append(name)
        else:
            raise AssertionError(f"accepted invalid input: {name}")

    x = [1, 0, 0, 0, 0, 0]
    prior = predict([], x)
    equal("prior_df", prior["df"], 2)
    equal("prior_loc", prior["loc"], [0, 0])
    equal("prior_scale", prior["scale"], np.eye(2))
    equal("prior_K", prior["posterior"]["K"], np.eye(6))
    equal("prior_M", prior["posterior"]["M"], np.zeros((6, 2)))
    equal("prior_Psi", prior["posterior"]["Psi"], np.eye(2))
    equal("prior_nu", prior["posterior"]["nu"], 3)
    equal("zero_feature_prior_scale", predict([], [0] * 6)["scale"], np.eye(2) / 2)

    one = predict([{"x": x, "y": [2, 3]}], x)
    equal("one_K", one["posterior"]["K"], np.diag([2, 1, 1, 1, 1, 1]))
    expected_mean = np.zeros((6, 2))
    expected_mean[0] = [1, 1.5]
    equal("one_M", one["posterior"]["M"], expected_mean)
    equal("one_Psi", one["posterior"]["Psi"], [[3, 3], [3, 5.5]])
    equal("one_scale", one["scale"], [[1.5, 1.5], [1.5, 2.75]])
    equal("one_nu_df", [one["posterior"]["nu"], one["df"]], [4, 3])

    # Independent per-coordinate scalar sums: diagonal conjugate posterior.
    history = [{"x": x, "y": [2, 3]},
               {"x": [0, 1, 0, 0, 0, 0], "y": [-1, 4]},
               {"x": x, "y": [4, -2]}]
    expected_k, expected_m, expected_psi = np.eye(6), np.zeros((6, 2)), np.eye(2)
    for a in range(2):
        for b in range(2):
            expected_psi[a, b] += sum(row["y"][a] * row["y"][b] for row in history)
    for coordinate in range(6):
        rows = [row for row in history if row["x"][coordinate] == 1]
        count = len(rows) + 1
        totals = [sum(row["y"][a] for row in rows) for a in range(2)]
        expected_k[coordinate, coordinate] = count
        for a in range(2):
            expected_m[coordinate, a] = totals[a] / count
            for b in range(2):
                expected_psi[a, b] -= totals[a] * totals[b] / count
    fitted = predict(history, [1, 2, 0, 0, 0, 0])
    equal("diagonal_K", fitted["posterior"]["K"], expected_k)
    equal("diagonal_M", fitted["posterior"]["M"], expected_m)
    equal("diagonal_Psi", fitted["posterior"]["Psi"], expected_psi)
    equal("diagonal_loc", fitted["loc"], expected_m[0] + 2 * expected_m[1])
    equal("diagonal_scale", fitted["scale"], (1 + 1 / 3 + 4 / 2) * expected_psi / 5)

    # Joint t logpdf minus marginal Y logpdf, using explicit 2x2 algebra.
    rng = np.random.default_rng(1847)
    for trial in range(40):
        sample = [{"x": rng.normal(size=6).tolist(), "y": rng.normal(size=2).tolist()}
                  for _ in range(trial % 8)]
        prediction = predict(sample, rng.normal(size=6).tolist())
        y, z = rng.normal(size=2)
        df, location, scale = prediction["df"], prediction["loc"], prediction["scale"]
        a, b, c = scale[0][0], scale[0][1], scale[1][1]
        determinant = a * c - b * b
        u, v = y - location[0], z - location[1]
        quadratic = (c * u * u - 2 * b * u * v + a * v * v) / determinant
        joint = (math.lgamma((df + 2) / 2) - math.lgamma(df / 2)
                 - math.log(df * math.pi) - math.log(determinant) / 2
                 - (df + 2) / 2 * math.log1p(quadratic / df))
        marginal = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
                    - math.log(df * math.pi * a) / 2
                    - (df + 1) / 2 * math.log1p(u * u / (df * a)))
        conditional = conditional_z(prediction, y)
        equal(f"t_factorization_{trial}", student_t_logpdf(z, **conditional), joint - marginal)
    equal("scalar_cauchy", student_t_logpdf(0, 1, 0, 1), -math.log(math.pi))
    equal("prior_conditional", list(conditional_z(prior, 2).values()), [3, 0, 2])

    for rho in (0, 0.7, -0.7):
        hypothesis = {"mean_y": 1, "mean_z": -2, "std_y": 2, "std_z": 3, "rho": rho}
        result = evidence(prior, hypothesis, 4, -0.5)
        expected_location = -2 + rho * 3 / 2 * 3
        expected_variance = 9 * (1 - rho * rho)
        expected_logpdf = -math.log(2 * math.pi * expected_variance) / 2 - (-0.5 - expected_location) ** 2 / (2 * expected_variance)
        equal(f"gaussian_location_{rho}", result["hypothesis_conditional"]["loc"], expected_location)
        equal(f"gaussian_std_{rho}", result["hypothesis_conditional"]["std"], math.sqrt(expected_variance))
        equal(f"gaussian_density_{rho}", result["hypothesis_logpdf"], expected_logpdf)
        equal(f"ratio_{rho}", result["log_ratio"], expected_logpdf - result["reference_logpdf"])
    unchanged = json.dumps(prior, sort_keys=True)
    evidence(prior, hypothesis, 2, 3)
    equal("evidence_does_not_update", json.dumps(prior, sort_keys=True) == unchanged, True)

    for name, bad in (("nan", float("nan")), ("inf", float("inf")), ("string", "1"), ("bool", True)):
        rejects(f"x_{name}", lambda bad=bad: predict([], [bad] * 6))
        rejects(f"y_{name}", lambda bad=bad: predict([{"x": x, "y": [bad, bad]}], x))
        rejects(f"observed_{name}", lambda bad=bad: evidence(prior, hypothesis, bad, 0))
    rejects("x_shape", lambda: predict([], [1, 2]))
    rejects("history_shape", lambda: predict({}, x))
    rejects("missing_y", lambda: predict([{"x": x}], x))
    rejects("response_shape", lambda: predict([{"x": x, "y": [1]}], x))
    rejects("overflow_posterior", lambda: predict([{"x": [1e308] * 6, "y": [1, 2]}], x))
    for key, bad in (("std_y", 0), ("std_z", -1), ("rho", 1), ("rho", -1), ("mean_z", float("nan"))):
        rejects(f"hypothesis_{key}_{bad}", lambda key=key, bad=bad: evidence(prior, {**hypothesis, key: bad}, 0, 0))
    for label, patch in (("df", {"df": 0}), ("scale", {"scale": [[1, 2], [2, 1]]}),
                         ("asymmetric", {"scale": [[1, 0.1], [0, 1]]}), ("loc", {"loc": [0, float("inf")]})):
        rejects(f"prediction_{label}", lambda patch=patch: conditional_z({**prior, **patch}, 0))
    rejects("scalar_scale", lambda: student_t_logpdf(0, 2, 0, 0))
    rejects("scalar_df", lambda: student_t_logpdf(0, -1, 0, 1))
    rejects("scalar_overflow", lambda: student_t_logpdf(1e308, 2, 0, 1e-308))
    # Strict serialization catches any accidental NumPy scalar or NaN output.
    json.dumps({"prediction": fitted, "evidence": result}, allow_nan=False)
    checks.append("json_serialization")
    output = {"status": "passed", "data": "generated_only", "numpy_version": np.__version__,
              "passed": len(checks), "failed": 0, "checks": checks}
    destination = Path(__file__).with_name("reference_unit_checks.json")
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("status", "passed", "failed", "data", "numpy_version")}))


if __name__ == "__main__":
    main()
