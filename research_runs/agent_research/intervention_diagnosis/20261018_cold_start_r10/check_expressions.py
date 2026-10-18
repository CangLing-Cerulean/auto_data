"""Synthetic-only checks; writes the adjacent JSON check receipt."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from coldstart_expressions import ExpressionError, measurement, recipe_weights


def main():
    checks = []

    def equal(name, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        checks.append({"name": name, "passed": True})

    def reject(name, operation):
        try:
            operation()
        except ExpressionError as exc:
            checks.append({"name": name, "passed": True, "error": str(exc)})
        else:
            raise AssertionError(f"{name}: expected ExpressionError")

    x = np.arange(7 * 5 * 3, dtype=float).reshape(7, 5, 3) / 10 - 5
    p = np.linspace(0, 1, 7)
    expressions = {
        "mean_channels(mean_time(x))": [w.mean(axis=0).mean() for w in x],
        "std_channels(std_time(x))": [w.std(axis=0).std() for w in x],
        "max_channels(last_time(x)-first_time(x))": [(w[-1]-w[0]).max() for w in x],
        "mean_time(channel(x,1))": [w[:, 1].mean() for w in x],
        "mean_channels(mean_time(diff_time(x)))": [np.diff(w, axis=0).mean() for w in x],
        "mean_time(max_channels(lagdiff(x,2)))": [(w[2:]-w[:-2]).max(axis=1).mean() for w in x],
        "mean_time(mean_channels(x))": [w.mean(axis=1).mean() for w in x],
        "safe_ratio(position,position)": p / (1 + abs(p)),
        "where(position<0.5,-position,position*2+1)": np.where(p < .5, -p, p*2+1),
        "log1p_abs(tanh(square(-position)))": np.log1p(abs(np.tanh(p*p))),
    }
    for expression, reference in expressions.items():
        equal(expression, measurement(expression, x, p), reference)
    for symbol in ("<", "<=", ">", ">="):
        equal(f"comparison {symbol}", measurement(f"where(position{symbol}0.5,1,0)", x, p),
              np.where({"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}[symbol](p, .5), 1, 0))
    invalid = ["x+position", "mean_time(x)+mean_channels(x)", "1", "x", "position<1",
               "x.__class__", "x[0]", "__import__('os')", "open('x')", "'text'", "True",
               "position/2", "position**2", "+position", "unknown", "abs(position,1)",
               "abs(v=position)", "[position]", "(lambda: 1)()", "position==0",
               "0<position<1", "channel(x,3)", "channel(x,-1)", "channel(x,1.0)",
               "channel(x,1+0)", "lagdiff(x,0)", "lagdiff(x,5)", "lagdiff(x,1.0)",
               "mean_time(position)", "sigmoid(position)", "exp_clip(position)",
               "where(position,1,0)", "1001*position", "1e309*position",
               "+".join(["position"] * 50)]
    for expression in invalid:
        reject(f"measurement rejects {expression}", lambda e=expression: measurement(e, x, p))
    reject("nonfinite input", lambda: measurement("position", x * np.nan, p))
    reject("nonfinite result", lambda: measurement("square(position)", x, np.full(7, 1e308)))
    reject("position shape", lambda: measurement("position", x, p[:, None]))
    reject("x shape", lambda: measurement("position", x[:, :, 0], p))
    reject("empty time", lambda: measurement("position", x[:, :0], p))
    reject("short diff", lambda: measurement("mean_channels(mean_time(diff_time(x)))", x[:, :1], p))
    reject("different time lengths", lambda: measurement("mean_channels(mean_time(x+diff_time(x)))", x, p))

    rp = np.linspace(0, 1, 512)
    q = np.linspace(-1000, 1000, 512)
    original = q.copy()
    weights, stats = recipe_weights("exp_clip(q1)+sigmoid(q1)", {"q1": q}, rp)
    raw = np.exp(np.clip(q, -4, 4)) + np.exp(-np.logaddexp(0, -q))
    equal("weight reference", weights, raw / raw.mean())
    equal("input preserved", q, original)
    equal("ESS", stats["ess"], weights.sum()**2 / np.dot(weights, weights))
    equal("sum of squares", stats["sum_weights_squared"], (weights**2).sum())
    equal("weight mean", stats["mean"], 1)
    equal("constant broadcast", recipe_weights("2", {}, rp)[0], np.ones(512))
    equal("256 positive accepted", recipe_weights("where(position<0.5,1,0)", {}, rp)[1]["nonzero"], 256)
    equal("stable sigmoid", recipe_weights("sigmoid(q1)", {"q1": q}, rp)[0],
          np.exp(-np.logaddexp(0, -q)) / np.exp(-np.logaddexp(0, -q)).mean())
    for expression in ["0", "-1", "where(position<0.4,1,0)",
                       "where(position<0.001,1000,0.001)", "square(q1)",
                       "x", "mean_time(q1)", "channel(q1,0)", "q9", "position<1"]:
        values = {"q1": np.full(512, 1e308)} if expression == "square(q1)" else {"q1": q}
        reject(f"recipe rejects {expression}", lambda e=expression, v=values: recipe_weights(e, v, rp))
    reject("unknown q key", lambda: recipe_weights("1", {"signal": q}, rp))
    reject("wrong q shape", lambda: recipe_weights("1", {"q1": q[:2]}, rp))
    reject("nonfinite q", lambda: recipe_weights("1", {"q1": q*np.nan}, rp))
    reject("small N", lambda: recipe_weights("1", {}, rp[:255]))
    receipt = {"status": "passed", "scope": "synthetic arrays only; no real data or model calls",
               "passed": len(checks), "failed": 0, "checks": checks}
    Path(__file__).with_name("expression_unit_checks.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {len(checks)} synthetic expression checks")


if __name__ == "__main__":
    main()
