"""Shared label completeness contract. Missing cells never reduce the denominator."""
from __future__ import annotations

import math

PROTOCOL_VERSION = "mature-labels-v1"


def issue_window(start, duration_ns, step, anchor):
    """Fixed grid-aligned issues strictly after a lifecycle transition."""
    first = anchor + ((start - anchor) // step + 1) * step
    count = (int(duration_ns) + step - 1) // step
    return range(first, first + count * step, step)


def expected_keys(rows, config, origins=None):
    origins = {r["issue_ns"] for r in rows} if origins is None else set(origins)
    return {(issue, issue + h * config.step_ns, variable, h)
            for issue in origins for h in range(1, config.horizon_steps + 1)
            for variable in config.variables}


def complete_labels(rows, config, origins=None):
    expected = expected_keys(rows, config, origins)
    observed = {(r["issue_ns"], r["target_ns"], r["variable"], r["horizon"]): r
                for r in rows}
    mature = {key for key, row in observed.items()
              if row.get("actual") is not None and math.isfinite(row["actual"])}
    return {"protocol_version": PROTOCOL_VERSION,
            "expected_count": len(expected),
            "labels_complete": bool(expected) and len(observed) == len(rows)
            and set(observed) == expected and mature == expected,
            "label_coverage": len(mature & expected) / len(expected) if expected else 0.0}
