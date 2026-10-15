"""Independent symbolic/enumeration checks, with no real data or training."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import minimize
import sympy as sp

from adam_risk_sampling import (NonsmoothAdam, adam_derivatives, allocate_signed,
    dlinear_scores, group_coefficients, quadratic_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("checks are immutable; do not overwrite a run")
    started = time.perf_counter()
    checks, enumerations = [], []

    def equal(name, actual, expected, atol=1e-12, rtol=2e-10):
        actual, expected = np.asarray(actual), np.asarray(expected)
        ok = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
        checks.append({"name": name, "pass": ok,
                       "max_abs_error": float(np.max(np.abs(actual - expected)))})

    x, a, b, c, d, e, eta = sp.symbols("x a b c d e eta", real=True)
    expression = -eta * (a + b*x) / (sp.sqrt(c + d*x*x) + e)
    symbolic = sp.lambdify((x, a, b, c, d, e, eta),
        [sp.diff(expression, x), sp.diff(expression, x, 2)], "numpy")
    for seed in (10301, 10302, 10303):
        rng = np.random.default_rng(seed)
        for step in (1, 501, 3001):
            theta, m, g = rng.normal(size=(3, 24))
            v = rng.uniform(0.2, 2, 24)
            u, first, second = adam_derivatives(theta, m, v, g, next_step=step)
            reference = symbolic(g, .9*m/(1-.9**step), .1/(1-.9**step),
                .999*v/(1-.999**step), .001/(1-.999**step), 1e-8, .001)
            equal(f"symbolic_first_{seed}_{step}", first, reference[0])
            equal(f"symbolic_second_{seed}_{step}", second, reference[1])
            # Direct update in the standard moment-then-bias-correction order.
            nm, nv = .9*m+.1*g, .999*v+.001*g*g
            reference_u = theta-.001*(nm/(1-.9**step))/(np.sqrt(nv/(1-.999**step))+1e-8)
            equal(f"direct_update_{seed}_{step}", u, reference_u)

        # Block normalization and ordering against an explicitly constructed H.
        batch_g = rng.normal(size=(6, 2, 4, 3))
        first, second, risk_gradient = rng.normal(size=(3, 2, 4, 3))
        feature = rng.normal(size=(7, 2, 4))
        moment = np.einsum("nci,ncj->cij", feature, feature)/7
        full_h = np.zeros((24, 24))
        for ch, h, i, j in itertools.product(range(2), range(3), range(4), range(4)):
            full_h[ch*12+i*3+h, ch*12+j*3+h] = 2*moment[ch, i, j]/6
        block = dlinear_scores(batch_g, first, second, risk_gradient, moment)
        dense = quadratic_scores(batch_g.reshape(6, -1), first.ravel(), second.ravel(),
            risk_gradient.ravel(), full_h)
        for key in block:
            equal(f"block_dense_{key}_{seed}", block[key], dense[key])

        # Fixed finite population: enumerate all 36 ordered size-2 batches.
        population = rng.normal(size=(6, 4))
        ids = np.array([0, 0, 1, 1, 2, 2])
        q = np.ones(3)/3
        theta, m, target = rng.normal(size=(3, 4))
        v = rng.uniform(.2, 2, 4)
        h_factor = rng.normal(size=(4, 4))
        hessian = h_factor.T @ h_factor
        mu = population.mean(axis=0)
        updated, first, second = adam_derivatives(theta, m, v, mu, next_step=501)
        risk_gradient = hessian @ (updated-target)
        scores = quadratic_scores(population, first, second, risk_gradient, hessian)
        coefficients = {key: group_coefficients(value, ids, 3)[0]
                        for key, value in scores.items()}
        risk_matrix = (first[:, None]*hessian*first[None, :]
                       + np.diag(risk_gradient*second))
        # SymPy differentiates the complete composed 4-dimensional risk.
        variables = sp.symbols("z0:4", real=True)
        us = sp.Matrix([float(theta[j])-.001*(.9*float(m[j])+.1*variables[j])
            /(1-.9**501)/(sp.sqrt((.999*float(v[j])+.001*variables[j]**2)
              /(1-.999**501))+1e-8)-float(target[j]) for j in range(4)])
        composed = (us.T*sp.Matrix(hessian)*us)[0]/2
        symbolic_hessian = sp.lambdify(variables, sp.hessian(composed, variables), "numpy")
        equal(f"composed_hessian_{seed}", risk_matrix, symbolic_hessian(*mu))

        policies = {"uniform": q}
        for name in ("gradient_norm", "jacobian_norm", "geometry", "full"):
            policies[name] = allocate_signed(coefficients[name], q)
        base_risk = .5*(updated-target)@hessian@(updated-target)
        for name, probability in policies.items():
            weights, batches, risks, quadratics = [], [], [], []
            for i, j in itertools.product(range(6), repeat=2):
                weight = probability[ids[i]]/2 * probability[ids[j]]/2
                gradient = (q[ids[i]]/probability[ids[i]]*population[i]
                            + q[ids[j]]/probability[ids[j]]*population[j])/2
                # Independent direct Adam update, not the derivative function.
                nm = .9*m+.1*gradient
                nv = .999*v+.001*gradient**2
                after = theta-.001*(nm/(1-.9**501))/(np.sqrt(nv/(1-.999**501))+1e-8)
                delta = gradient-mu
                weights.append(weight)
                batches.append(gradient)
                risks.append(.5*(after-target)@hessian@(after-target))
                quadratics.append(base_risk+(risk_gradient*first)@delta
                                  + .5*delta@risk_matrix@delta)
            weights, batches = np.asarray(weights), np.asarray(batches)
            covariance = (batches-mu).T @ (weights[:, None]*(batches-mu))
            formula_cov = sum(q[k]**2/probability[k]
                * (population[ids == k].T @ population[ids == k]/2)
                for k in range(3))/2-np.outer(mu, mu)/2
            proxy = base_risk+(sum(coefficients["full"]/probability)-mu@risk_matrix@mu)/4
            equal(f"probability_sum_{seed}_{name}", weights.sum(), 1.)
            equal(f"mean_invariance_{seed}_{name}", weights@batches, mu)
            equal(f"covariance_identity_{seed}_{name}", covariance, formula_cov)
            equal(f"risk_proxy_identity_{seed}_{name}", weights@quadratics, proxy)
            enumerations.append({"seed": seed, "policy": name, "P": probability.tolist(),
                "exact_conditional_risk": float(weights@risks), "proxy": float(proxy),
                "proxy_error": float(proxy-weights@risks),
                "full_coefficients": coefficients["full"].tolist()})

    q = np.array([.2, .3, .5])
    solver_cases = [[1, 2, 3], [1, 1e-8, 3], [-1, 2, 0],
                    [-1, -2, -3], [0, -2, 0], [0, 0, 0]]
    mesh = np.array([[i/200, j/200, 1-(i+j)/200]
                    for i in range(201) for j in range(201-i)])
    grid = .5*q+.5*mesh
    for index, case in enumerate(solver_cases):
        coefficients = np.asarray(case, dtype=float)
        probability = allocate_signed(coefficients, q)
        optimum = float(sum(coefficients/probability))
        grid_best = float(np.min((coefficients/grid).sum(axis=1)))
        scipy_values, scipy_success = [], []
        for start in [q, *(.5*q+.5*np.eye(3))]:
            result = minimize(lambda p: sum(coefficients/p), start,
                jac=lambda p: -coefficients/p**2,
                bounds=[(lower, 1.) for lower in .5*q],
                constraints={"type": "eq", "fun": lambda p: p.sum()-1,
                             "jac": lambda p: np.ones(3)},
                method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
            scipy_values.append(float(result.fun))
            scipy_success.append(bool(result.success))
        checks.append({"name": f"signed_solver_{index}",
            "pass": bool(optimum <= grid_best+1e-10
                and abs(optimum-min(scipy_values)) <= 1e-8
                and all(scipy_success) and abs(probability.sum()-1) < 1e-12
                and np.all(probability >= .5*q-1e-15)),
            "a": case, "P": probability.tolist(), "objective": optimum,
            "grid_best": grid_best, "scipy_values": scipy_values,
            "scipy_success": scipy_success})
    try:
        adam_derivatives(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), next_step=1)
    except NonsmoothAdam:
        checks.append({"name": "nonsmooth_domain_rejected", "pass": True})
    else:
        checks.append({"name": "nonsmooth_domain_rejected", "pass": False})
    root = Path(__file__).resolve().parents[1]
    inputs = [Path(__file__), Path(__file__).with_name("adam_risk_sampling.py"),
              args.out.parent / "DESIGN.md", args.out.parent / "CHECK_PLAN.json"]
    result = {"status": "PASS" if all(x["pass"] for x in checks) else "FAIL",
        "checks": checks, "exact_enumeration_descriptive_only": enumerations,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sympy": sp.__version__},
        "elapsed_seconds": time.perf_counter()-started,
        "new_full_trainings": 0, "real_arrays_accessed": False,
        "inputs": [{"path": str(p.relative_to(root)).replace("\\", "/"),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "failed": [x["name"] for x in checks if not x["pass"]]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
