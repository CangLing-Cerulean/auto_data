"""Restricted, axis-aware expressions for cold-start window measurements.

No fitting or file access occurs here. Callers supply already standardized q's.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping

import numpy as np


class ExpressionError(ValueError):
    """An expression, its inputs, or its output violates the public contract."""


@dataclass(frozen=True)
class Value:
    data: np.ndarray
    axes: tuple[str, ...]


def _array(data, name: str) -> np.ndarray:
    try:
        result = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExpressionError(f"{name} must be a finite numeric array") from exc
    if not np.all(np.isfinite(result)):
        raise ExpressionError(f"{name} must contain only finite values")
    return result


def _position(position) -> np.ndarray:
    result = _array(position, "position")
    if result.ndim != 1 or result.size == 0:
        raise ExpressionError("position must have nonempty shape [N]")
    return result


def _common_axes(*values: Value) -> tuple[str, ...]:
    non_scalar = [value for value in values if value.axes]
    if not non_scalar:
        return ()
    first = non_scalar[0]
    if any(value.axes != first.axes or value.data.shape != first.data.shape
           for value in non_scalar[1:]):
        raise ExpressionError("non-scalar operands must have identical axes and shapes")
    return first.axes


class _Interpreter:
    def __init__(self, variables: Mapping[str, Value], recipe: bool):
        self.variables = variables
        self.recipe = recipe

    def run(self, expression: str) -> Value:
        if not isinstance(expression, str):
            raise ExpressionError("expression must be a string")
        # Bound parsing as well as evaluation; valid 128-node expressions fit easily.
        if len(expression) > 16384:
            raise ExpressionError("expression is too long")
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise ExpressionError("invalid expression syntax") from exc
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ExpressionError("expression exceeds 128 AST nodes")
        try:
            with np.errstate(all="ignore"):
                return self.visit(tree.body)
        except ExpressionError:
            raise
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ExpressionError(f"expression evaluation failed: {exc}") from exc

    def visit(self, node: ast.AST) -> Value:
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ExpressionError("only finite numeric constants are allowed")
            if abs(node.value) > 1000 or not np.isfinite(node.value):
                raise ExpressionError("constants must be finite with absolute value <= 1000")
            return Value(np.asarray(node.value, dtype=np.float64), ())
        if isinstance(node, ast.Name):
            if node.id not in self.variables:
                raise ExpressionError(f"unknown variable: {node.id}")
            return self.variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self.visit(node.operand)
            return Value(-value.data, value.axes)
        if isinstance(node, ast.BinOp):
            operations = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply}
            operation = operations.get(type(node.op))
            if operation is None:
                raise ExpressionError("only +, -, and * binary operators are allowed")
            left, right = self.visit(node.left), self.visit(node.right)
            return Value(operation(left.data, right.data), _common_axes(left, right))
        if isinstance(node, ast.Compare):
            operations = {ast.Lt: np.less, ast.LtE: np.less_equal,
                          ast.Gt: np.greater, ast.GtE: np.greater_equal}
            if len(node.ops) != 1 or type(node.ops[0]) not in operations:
                raise ExpressionError("only single <, <=, >, >= comparisons are allowed")
            left, right = self.visit(node.left), self.visit(node.comparators[0])
            return Value(operations[type(node.ops[0])](left.data, right.data),
                         _common_axes(left, right))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ExpressionError("calls must use an allowed function name without keywords")
            return self.call(node.func.id, node.args)
        raise ExpressionError(f"forbidden syntax: {type(node).__name__}")

    def call(self, name: str, nodes: list[ast.AST]) -> Value:
        unary = {"abs": np.abs, "tanh": np.tanh, "square": np.square,
                 "log1p_abs": lambda v: np.log1p(np.abs(v))}
        if self.recipe:
            unary.update({"sigmoid": lambda v: np.exp(-np.logaddexp(0, -v)),
                          "exp_clip": lambda v: np.exp(np.clip(v, -4, 4))})
        reductions = {"mean_time": ("time", np.mean), "std_time": ("time", np.std),
                      "mean_channels": ("channel", np.mean),
                      "std_channels": ("channel", np.std),
                      "max_channels": ("channel", np.max)}
        allowed = set(unary) | {"safe_ratio", "where"}
        if not self.recipe:
            allowed |= set(reductions) | {"first_time", "last_time", "diff_time",
                                          "channel", "lagdiff"}
        if name not in allowed:
            raise ExpressionError(f"function is not allowed: {name}")
        expected = 3 if name == "where" else 2 if name in {"safe_ratio", "channel", "lagdiff"} else 1
        if len(nodes) != expected:
            raise ExpressionError(f"{name} requires {expected} arguments")
        value = self.visit(nodes[0])
        if name in unary:
            return Value(unary[name](value.data), value.axes)
        if name == "safe_ratio":
            denominator = self.visit(nodes[1])
            return Value(value.data / (1 + np.abs(denominator.data)),
                         _common_axes(value, denominator))
        if name == "where":
            left, right = self.visit(nodes[1]), self.visit(nodes[2])
            if value.data.dtype.kind != "b":
                raise ExpressionError("where condition must be a comparison")
            return Value(np.where(value.data, left.data, right.data),
                         _common_axes(value, left, right))
        axis_name = "channel" if name == "channel" else reductions[name][0] if name in reductions else "time"
        if axis_name not in value.axes:
            raise ExpressionError(f"{name} requires the {axis_name} axis")
        axis = value.axes.index(axis_name)
        reduced_axes = value.axes[:axis] + value.axes[axis + 1:]
        if name in reductions:
            return Value(reductions[name][1](value.data, axis=axis), reduced_axes)
        if name in {"first_time", "last_time"}:
            return Value(np.take(value.data, 0 if name == "first_time" else -1, axis=axis), reduced_axes)
        if name in {"channel", "lagdiff"}:
            index_node = nodes[1]
            if not isinstance(index_node, ast.Constant) or type(index_node.value) is not int:
                raise ExpressionError(f"{name} index must be a literal integer")
            index = index_node.value
            if abs(index) > 1000:
                raise ExpressionError("integer constants must have absolute value <= 1000")
            if name == "channel":
                if not 0 <= index < value.data.shape[axis]:
                    raise ExpressionError("channel index is out of range")
                return Value(np.take(value.data, index, axis=axis), reduced_axes)
            if not 1 <= index < value.data.shape[axis]:
                raise ExpressionError("lagdiff requires 1 <= k < time length")
        else:
            index = 1
            if value.data.shape[axis] < 2:
                raise ExpressionError("diff_time requires time length >= 2")
        later = [slice(None)] * value.data.ndim
        earlier = later.copy()
        later[axis], earlier[axis] = slice(index, None), slice(None, -index)
        return Value(value.data[tuple(later)] - value.data[tuple(earlier)], value.axes)


def measurement(expression: str, x, position) -> np.ndarray:
    """Evaluate a measurement on finite x[N,L,C], returning a fresh float64 [N]."""
    positions = _position(position)
    windows = _array(x, "x")
    if windows.ndim != 3 or windows.shape[0] != positions.size or min(windows.shape) < 1:
        raise ExpressionError("x must have nonempty shape [N,L,C] matching position")
    result = _Interpreter({"x": Value(windows, ("window", "time", "channel")),
                           "position": Value(positions, ("window",))}, False).run(expression)
    if result.axes != ("window",) or result.data.shape != positions.shape:
        raise ExpressionError("measurement must return exactly the window axis [N]")
    if result.data.dtype.kind == "b" or not np.all(np.isfinite(result.data)):
        raise ExpressionError("measurement must return finite numeric floats")
    return np.array(result.data, dtype=np.float64, copy=True)


def recipe_weights(expression: str, measurements: Mapping[str, object], position) -> tuple[np.ndarray, dict]:
    """Return mean-one weights and explicit raw/normalized weight statistics.

    Reject invalid weights without clipping or repair. Neither inputs nor q's
    are modified; q standardization belongs to the caller.
    """
    positions = _position(position)
    if not isinstance(measurements, Mapping):
        raise ExpressionError("measurements must be a mapping of q1 through q8")
    variables = {"position": Value(positions, ("window",))}
    for name, data in measurements.items():
        if name not in {f"q{i}" for i in range(1, 9)}:
            raise ExpressionError(f"invalid measurement variable: {name}")
        values = _array(data, name)
        if values.shape != positions.shape:
            raise ExpressionError(f"{name} must have shape [N] matching position")
        variables[name] = Value(values, ("window",))
    result = _Interpreter(variables, True).run(expression)
    if result.axes not in {(), ("window",)} or result.data.dtype.kind == "b":
        raise ExpressionError("recipe must return numeric scalar or window weights")
    weights = np.full(positions.size, result.data.item(), dtype=np.float64) if not result.axes else np.array(result.data, copy=True)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ExpressionError("weights must be finite and nonnegative")
    nonzero = int(np.count_nonzero(weights))
    # Scaled normalization avoids overflow in the mean of otherwise finite weights.
    maximum = float(np.max(weights))
    if maximum <= 0:
        raise ExpressionError("weights must have positive mean")
    if nonzero < 256:
        raise ExpressionError("weights require at least 256 positive windows")
    scaled = weights / maximum
    scaled_mean = float(np.mean(scaled))
    normalized = scaled / scaled_mean
    if np.max(normalized) > 20:
        raise ExpressionError("normalized maximum weight must be <= 20")
    sum_weights = float(np.sum(normalized))
    sum_squared = float(np.dot(normalized, normalized))
    statistics = {"n": int(weights.size), "nonzero": nonzero,
                  "raw_mean": maximum * scaled_mean, "raw_max": maximum,
                  "mean": float(np.mean(normalized)), "max": float(np.max(normalized)),
                  "sum_weights": sum_weights, "sum_weights_squared": sum_squared,
                  "squared_sum_weights": sum_weights * sum_weights,
                  "ess": sum_weights * sum_weights / sum_squared}
    return normalized, statistics
