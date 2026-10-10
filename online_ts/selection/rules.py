"""Bounded expression grammar; no Python evaluation or filesystem arguments."""
import math

import numpy as np

from ..config import GateError, digest
from .dataset import FEATURES

OPS = ("lt", "le", "gt", "ge")
QUANTILES = (.1, .25, .5, .75, .9)
HISTORICAL_JACCARD_LIMIT = .95


def compile_rule(rule, data, config):
    nodes = 0

    def visit(value):
        nonlocal nodes
        nodes += 1
        if nodes > config.max_rule_nodes or not isinstance(value, dict):
            raise GateError("RULE_COMPLEXITY")
        kind = value.get("kind")
        if kind == "all" and set(value) == {"kind"}:
            return np.ones(len(data.starts), dtype=bool)
        if kind in {"and", "or"} and set(value) == {"kind", "children"}:
            children = value["children"]
            if not isinstance(children, list) or not 2 <= len(children) <= 3:
                raise GateError("RULE_CHILDREN")
            masks = [visit(c) for c in children]
            return np.logical_and.reduce(masks) if kind == "and" else np.logical_or.reduce(masks)
        if kind == "not" and set(value) == {"kind", "child"}:
            return ~visit(value["child"])
        if kind != "compare" or set(value) != {"kind", "feature", "op", "value", "scale"}:
            raise GateError("RULE_SCHEMA")
        feature, op, threshold, scale = (value[k] for k in ("feature", "op", "value", "scale"))
        if (feature not in FEATURES or op not in OPS or type(threshold) not in {int, float}
                or not math.isfinite(threshold) or scale not in {"quantile", "literal"}):
            raise GateError("RULE_COMPARISON")
        values = data.features[feature]
        if scale == "quantile":
            if not 0 <= threshold <= 1:
                raise GateError("RULE_QUANTILE")
            threshold = float(np.quantile(values, threshold))
        elif not float(values.min()) <= threshold <= float(values.max()):
            raise GateError("RULE_THRESHOLD_RANGE")
        return {"lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal}[op](values, threshold)

    return visit(rule)


def mask_hash(mask):
    return digest({"length": len(mask), "packed_hex": np.packbits(mask).tobytes().hex()})


def check_mask(mask, data, config, *, historical=True):
    if mask.dtype != np.bool_ or mask.shape != data.starts.shape:
        raise GateError("MASK_SCHEMA")
    issues, groups = [], []
    if mask.mean() < config.min_retention:
        issues.append("TOTAL_RETENTION")
    for i, name in enumerate(data.arrays["group_names"]):
        members = data.groups == i
        total, retained = int(members.sum()), int(mask[members].sum())
        groups.append({"group": str(name), "eligible": total, "retained": retained})
        if retained < math.ceil(total * config.min_group_retention):
            issues.append("GROUP_COVERAGE:" + str(name))
        if total <= config.rare_group_size and retained != total:
            issues.append("RARE_GROUP:" + str(name))
    overlaps = []
    if historical:
        for i, ref in enumerate(data.task.get("historical_masks", [])):
            domain = data.arrays[f"historical_domain_{i}"]
            old = data.arrays[f"historical_removed_{i}"][domain]
            new = ~mask[domain]
            union = int((old | new).sum())
            overlap = float((old & new).sum() / union) if union else 0.
            overlaps.append({**ref, "common_windows": int(domain.sum()), "removed_jaccard": overlap})
            if union and overlap >= HISTORICAL_JACCARD_LIMIT:
                issues.append("HISTORICAL_MASK_REPEAT:" + ref["path"])
    return {"valid": not issues, "issues": issues, "eligible": len(mask), "retained": int(mask.sum()),
            "retention": float(mask.mean()), "mask_sha256": mask_hash(mask),
            "groups": groups, "historical_overlap": overlaps}


def random_rule(rng):
    # A declared search distribution, not random row deletion. All agents have
    # the same grammar and bounds; this baseline has no adaptive rule decisions.
    def leaf():
        return {"kind": "compare", "feature": str(rng.choice(FEATURES)),
                "op": str(rng.choice(OPS)), "scale": "quantile", "value": float(rng.choice(QUANTILES))}
    shape = str(rng.choice(("single", "and", "or")))
    return leaf() if shape == "single" else {"kind": shape, "children": [leaf(), leaf()]}


def matched_controls(mask, data, seed):
    rng = np.random.default_rng(seed)
    count = np.zeros(len(mask), bool)
    count[rng.choice(len(mask), int(mask.sum()), replace=False)] = True
    quota = np.zeros(len(mask), bool)
    for group in np.unique(data.groups):
        ids = np.flatnonzero(data.groups == group)
        quota[rng.choice(ids, int(mask[ids].sum()), replace=False)] = True
    return {"same_count": count, "same_group_quota": quota}
