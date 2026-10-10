"""Actionable, structured feedback without rewriting a model's arguments."""
from collections.abc import Mapping
from dataclasses import MISSING, fields

from ...config import GateError


class ToolArgumentError(GateError):
    def __init__(self, code, issues):
        super().__init__(code)
        self.validation_errors = issues


def validate_wait_shape(args):
    from ..waiting import WAIT_SCHEMA
    missing, unexpected = sorted(set(WAIT_SCHEMA) - set(args)), sorted(set(args) - set(WAIT_SCHEMA))
    issues = []
    if missing or unexpected:
        issues.append({"path": "arguments", "missing_fields": missing, "unexpected_fields": unexpected,
                       "expected_fields": WAIT_SCHEMA})
    if "predicate" in args and not isinstance(args["predicate"], dict):
        issues.append({"path": "arguments.predicate", "expected": "JSON object", "schema": WAIT_SCHEMA["predicate"]})
    if issues:
        code = "AGENT_TOOL_ARGUMENTS" if missing or unexpected else "AGENT_WAIT_PREDICATE_MUST_BE_JSON_OBJECT"
        raise ToolArgumentError(code, issues)


def validate_diagnosis(candidates):
    from dataclasses import asdict
    pairs = [{"candidates": [i, j], "changed_fields": [k for k, value in asdict(a).items() if value != asdict(b)[k]]}
             for i, a in enumerate(candidates) for j, b in enumerate(candidates) if j > i]
    if not any(len(pair["changed_fields"]) == 1 for pair in pairs):
        raise ToolArgumentError("DIAGNOSIS_CONTROL_REQUIRED", [{"path": "arguments.candidates",
            "requirement": "At least one pair differs in exactly one expanded Candidate field; champion inclusion is not required.",
            "pairs": pairs, "purpose_scope": "optimization permits multi-field comparisons but does not establish single-factor attribution"}])


def object_issue(cls, value, path):
    """Describe dataclass shape from the actual parser's field definitions."""
    definitions = fields(cls)
    required = [f.name for f in definitions if f.default is MISSING and f.default_factory is MISSING]
    allowed = [f.name for f in definitions]
    if not isinstance(value, Mapping):
        return {"path": path, "expected": "object", "required_fields": required,
                "allowed_fields": allowed}
    missing = sorted(set(required) - set(value))
    unexpected = sorted(set(value) - set(allowed))
    if missing or unexpected:
        return {"path": path, "missing_fields": missing, "unexpected_fields": unexpected,
                "required_fields": required, "allowed_fields": allowed}
    return None


def parse_candidates(values, config):
    """Validate every offered candidate before any experiment side effect."""
    from ...models import Candidate
    from ...experiments import FeatureSpec
    candidates, issues = [], []
    first_code = None
    for index, value in enumerate(values):
        path = f"arguments.candidates[{index}]"
        try:
            candidate = Candidate.from_dict(value)
            candidate.validate(config)
            candidates.append(candidate)
        except (GateError, TypeError, ValueError, OverflowError) as error:
            code = str(error) if isinstance(error, GateError) else "AGENT_TOOL_ARGUMENTS"
            first_code = first_code or code
            detail = object_issue(Candidate, value, path)
            if detail is None and isinstance(value.get("features"), (list, tuple)):
                for feature_index, feature in enumerate(value["features"][:16]):
                    detail = object_issue(FeatureSpec, feature, f"{path}.features[{feature_index}]")
                    if detail:
                        break
            if detail is None and code == "FEATURE_OPERATION_NOT_AUTHORIZED":
                detail = {"path": path + ".features", "allowed_operations": list(config.search.feature_operations),
                    "base_inputs": list(config.variables), "requirement":
                    "features adds optional transforms only. All original variables are already inputs; features=[] is valid for ordinary ridge."}
            if detail is None and code == "SEASONAL_PERIOD":
                detail = {"path": path + ".period", "required_value": config.search.seasonal_period,
                    "supplied_value": value.get("period"), "default_if_omitted": Candidate.__dataclass_fields__["period"].default,
                    "requirement": "Set period explicitly to the task seasonal period; it must not exceed input_steps."}
            issues.append({"code": code, **(detail or {"path": path}),
                           "constraints_ref": "experiment_capabilities"})
    if issues:
        raise ToolArgumentError(first_code, issues)
    return candidates
