from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from importlib.resources import files

from .config import CHRONOS_MODEL_REVISIONS, GateError, TaskConfig, canonical, digest
from .llm_transport import LLMCallTrace, request_chat_bytes
from .context import build_context, fit_messages
from .models import Candidate
from .execution import probe_training_targets
from .remediation import AgentContext, FEEDBACK_SCHEMA_VERSION, FeedbackProposal


FEEDBACK_PROMPT_VERSION = "feedback-repair-v3"


def catalog_only(config: TaskConfig) -> bool:
    return not (config.search.parameter_bounds or config.search.feature_operations or config.foundation.enabled)


def planner_prompt_version(config: TaskConfig) -> str:
    return "catalog-planner-v2" if catalog_only(config) else "experiment-planner-v3"


def catalog_feedback_observation(context: AgentContext):
    """Request-scoped aliases remove repeated hashes, not evidence or constraints."""
    from statistics import mean
    catalog = context.allowed_changes["candidate_catalog"]
    candidates = {item["candidate_ref"]: Candidate.from_dict(item["candidate"]) for item in catalog}
    refs = {candidate.candidate_id: ref for ref, candidate in candidates.items()}
    history = list(context.history)
    issue_refs = {f"i{i}": issue["issue_id"] for i, issue in enumerate(context.gate_issues)}
    evidence_refs = {f"e{i}": ref for i, ref in enumerate(context.evidence_refs)}
    parent = context.comparisons.get("parent")
    best = context.comparisons.get("best_so_far")
    def history_ref(candidate_id):
        return next((f"h{i}" for i, item in enumerate(history) if item.get("candidate_id") == candidate_id), None)
    tried = set(context.current_plan.get("candidate_ids", []))
    observation = {
        "plan_version": context.plan_version,
        "candidate_columns": ["family", "input_steps", "history_steps", "alpha", "period"],
        "candidates": {ref: [c.family, c.input_steps, c.history_steps, c.alpha, c.period]
                       for ref, c in candidates.items()},
        "current_candidate": refs.get(context.outcome["candidate_id"], context.current_plan["actual_config"]),
        "history_columns": ["candidate_ref", "status", "score", "score_std"],
        "history": {f"h{i}": [refs.get(item.get("candidate_id")), item["status"],
                                (item.get("metrics") or {}).get("score"),
                                (item.get("metrics") or {}).get("score_std")]
                    for i, item in enumerate(history)},
        "issues": {ref: {key: context.gate_issues[i][key] for key in
                         ("code", "observed_value", "threshold", "direction")}
                   for i, ref in enumerate(issue_refs)},
        "evidence_refs": list(evidence_refs),
        "evidence_scope": context.outcome["evidence_scope"],
        "evaluation_protocol": context.outcome["evaluation_protocol"],
        "comparison": {"parent": history_ref(parent["candidate_id"]) if parent else None,
                       "best": history_ref(best["candidate_id"]) if best else None,
                       "best_candidate_ref": refs.get(best["candidate_id"]) if best else None,
                       "best_score": best.get("score") if best else None,
                       "comparable": context.comparisons.get("comparable"),
                       "baseline_mean_mae": {name: (item["normalized_mae"] if "normalized_mae" in item
                                                    else mean(item["normalized_mae_by_fold"]))
                                             for name, item in context.comparisons.get("baselines", {}).items()}},
        "allowed_changes": {"candidate_refs": [ref for ref, c in candidates.items() if c.candidate_id not in tried],
                            "training_targets": context.allowed_changes["training_targets"]},
        "remaining_budget": context.remaining_budget,
    }
    return observation, issue_refs, evidence_refs


class PlannerResponseError(GateError):
    """A billed model response failed deterministic parsing or admission."""

    def __init__(self, code: str, tokens: int, response_fingerprint: str):
        super().__init__(code)
        self.tokens = tokens
        self.response_fingerprint = response_fingerprint


def _billed_response_error(code: str, tokens, response) -> PlannerResponseError | None:
    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0:
        return PlannerResponseError(
            code, tokens, "response-" + digest(response)[:24])
    return None


def feedback_prompt() -> str:
    prompt = files("online_ts").joinpath("prompts/feedback_repair_v3.txt").read_text(encoding="utf-8")
    if not prompt.startswith("PROMPT_VERSION=" + FEEDBACK_PROMPT_VERSION):
        raise GateError("FEEDBACK_PROMPT_VERSION")
    return prompt


def proposal_fingerprint(proposal: dict) -> str:
    """Hash only executable semantics after schema defaults/type normalization."""
    raw = proposal.get("candidates", []) if isinstance(proposal, dict) else []
    # Candidate order is executable workflow state: reordering changes which
    # attempt runs first and therefore must not collapse to the same identity.
    candidates = [Candidate.from_dict(item).candidate_id for item in raw]
    return digest({"snapshot_sha": proposal.get("snapshot_sha"), "candidates": candidates,
                   "training_target": proposal.get("training_target")})


def allowed_candidates(config: TaskConfig) -> list[Candidate]:
    result = []
    # Rotate recipes over diagonal D/L pairs, then cover the remaining cross product.
    recipes = [("ridge", config.search.ridge_alphas[0]), ("last_value", 1.0), ("seasonal_naive", 1.0)] + [("ridge", a) for a in config.search.ridge_alphas[1:]]
    if "patchtst" in config.search.model_families:
        recipes.insert(1, ("patchtst", 1.0))
    recipes = [recipe for recipe in recipes if recipe[0] in config.search.model_families]
    histories, inputs = config.search.histories, config.search.inputs
    pairs = []
    for i in range(max(len(histories), len(inputs))):
        pair = (histories[i % len(histories)], inputs[i % len(inputs)])
        if pair not in pairs:
            pairs.append(pair)
    pairs.extend((h, l) for h in histories for l in inputs if (h, l) not in pairs)
    for round_index in range(len(pairs)):
        for recipe_index, (family, alpha) in enumerate(recipes):
            history, length = pairs[(round_index + recipe_index) % len(pairs)]
            extra = ({"model_id": "patchtst", "adaptation_mode": "from_scratch",
                      "batch_size": config.patchtst.batch_size, "num_steps": config.patchtst.num_steps,
                      "learning_rate": config.patchtst.learning_rate} if family == "patchtst" else {})
            candidate = Candidate(family, length, history, alpha, config.search.seasonal_period if family == "seasonal_naive" else 1, **extra)
            try:
                candidate.validate(config)
            except GateError:
                continue
            result.append(candidate)
    if config.foundation.enabled:
        pair_index = 0
        for model_id in config.foundation.model_ids:
            if "zero_shot" in config.foundation.adaptation_modes:
                history, length = pairs[pair_index % len(pairs)]
                pair_index += 1
                candidate = Candidate(
                    "chronos2", length, history,
                    model_id=model_id, revision=CHRONOS_MODEL_REVISIONS[model_id],
                    adaptation_mode="zero_shot", quantile_levels=config.foundation.quantile_levels,
                    batch_size=config.foundation.batch_sizes[0],
                    learning_rate=1e-5, num_steps=2, lora_rank=4,
                )
                candidate.validate(config)
                result.append(candidate)
        if "lora" in config.foundation.adaptation_modes:
            recipes_count = max(len(config.foundation.learning_rates), len(config.foundation.num_steps),
                                len(config.foundation.lora_ranks), len(config.foundation.batch_sizes))
            produced = 0
            for index in range(recipes_count * max(1, len(config.foundation.model_ids))):
                if produced >= config.foundation.max_lora_candidates:
                    break
                model_id = config.foundation.model_ids[index % len(config.foundation.model_ids)]
                history, length = pairs[(pair_index + index) % len(pairs)]
                candidate = Candidate(
                    "chronos2", length, history,
                    model_id=model_id, revision=CHRONOS_MODEL_REVISIONS[model_id],
                    adaptation_mode="lora", quantile_levels=config.foundation.quantile_levels,
                    batch_size=config.foundation.batch_sizes[index % len(config.foundation.batch_sizes)],
                    learning_rate=config.foundation.learning_rates[index % len(config.foundation.learning_rates)],
                    num_steps=config.foundation.num_steps[index % len(config.foundation.num_steps)],
                    lora_rank=config.foundation.lora_ranks[index % len(config.foundation.lora_ranks)],
                )
                try:
                    candidate.validate(config)
                except GateError:
                    continue
                result.append(candidate)
                produced += 1
    if not result:
        raise GateError("NO_FEASIBLE_CANDIDATES")
    return result


def candidate_pool(config: TaskConfig) -> list[Candidate]:
    """The finite consideration pool is independent of the execution allowance."""
    base = allowed_candidates(config)
    limit = config.planner.candidate_pool_size
    result = base[:max(1, limit // 2)] if not catalog_only(config) else base[:limit]
    bounds = config.search.parameter_bounds
    if "ridge" in config.search.model_families and bounds:
        from dataclasses import replace
        ridge = next((c for c in base if c.family == "ridge"), None)
        if ridge is not None:
            for fraction in (0.5, 0.25, 0.75):
                changes = {}
                for name, (low, high) in bounds.items():
                    changes[name] = ((low * (high / low) ** fraction) if name == "alpha"
                                     else int(round(low + fraction * (high - low))))
                candidate = replace(ridge, **changes)
                try:
                    candidate.validate(config)
                except GateError:
                    continue
                if candidate not in result:
                    result.append(candidate)
    result.extend(c for c in base if c not in result)
    return result[:limit]


def _capability_observation(config):
    from .experiments import capabilities
    value = capabilities(config)
    value["contract_ref"] = "contract"
    value["feature_schema"] = {"operation": "rolling_std", "source_field": "target:0", "window": 2}
    if not config.foundation.enabled:
        for name in ("locked_models", "adaptation_modes", "foundation_parameters"):
            value.pop(name)
    return value


def _expand_spec(value, config, evidence_aliases, *, feedback_id=None, parent_aliases=None):
    """Expand only issued aliases; all other fields still reach strict compilation."""
    if not isinstance(value, dict) or value.get("contract_ref") != "contract":
        raise GateError("EXPERIMENT_CONTRACT_VERSION")
    spec = dict(value)
    refs = spec.get("evidence_refs")
    if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in evidence_aliases for ref in refs):
        raise GateError("EXPERIMENT_EVIDENCE")
    spec["contract_ref"] = config.version
    spec["evidence_refs"] = [evidence_aliases[ref] for ref in refs]
    if spec.get("feedback_id") is not None:
        if feedback_id is None or spec["feedback_id"] != "feedback":
            raise GateError("EXPERIMENT_FEEDBACK_VERSION")
        spec["feedback_id"] = feedback_id
    elif feedback_id is not None:
        raise GateError("EXPERIMENT_FEEDBACK_VERSION")
    if spec.get("parent_candidate_id") is not None:
        if spec["parent_candidate_id"] not in (parent_aliases or {}):
            raise GateError("EXPERIMENT_PARENT")
        spec["parent_candidate_id"] = parent_aliases[spec["parent_candidate_id"]]
    return spec


def _read_response(transport, request, config, trace, on_request, cancelled):
    if on_request is not None:
        on_request(trace)
    trace.emit(force=True)
    if transport is request_chat_bytes:
        payload = transport(request, config.planner.timeout_seconds, trace=trace, cancelled=cancelled)
    else:
        trace.mark("request_start_ms")
        payload = transport(request, config.planner.timeout_seconds)
        trace.data["transport_kind"] = "injected_transport"
    started = time.monotonic()
    response = json.loads(payload)
    trace.data["usage"] = response.get("usage")
    if response["choices"][0].get("finish_reason") == "length":
        raise GateError("PLANNER_RESPONSE_INCOMPLETE")
    text = response["choices"][0]["message"]["content"]
    trace.data["model_response"] = text
    decision = json.loads(text)
    trace.mark("json_parsed_ms")
    trace.data["json_parse_duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    trace.data["complete_json_arrival_ms"] = trace.data.get("last_answer_content_ms", trace.data["json_parsed_ms"])
    return response, decision


def _candidate_observation(candidate: Candidate, candidate_ref: str) -> dict:
    value = {"candidate_ref": candidate_ref, "family": candidate.family,
             "input_steps": candidate.input_steps, "history_steps": candidate.history_steps}
    if candidate.family == "ridge":
        value["alpha"] = candidate.alpha
    elif candidate.family == "seasonal_naive":
        value["period"] = candidate.period
    elif candidate.family == "chronos2":
        value.update(model_id=candidate.model_id, revision=candidate.revision,
                     adaptation_mode=candidate.adaptation_mode,
                     quantile_levels=list(candidate.quantile_levels), batch_size=candidate.batch_size,
                     learning_rate=candidate.learning_rate, num_steps=candidate.num_steps,
                     lora_rank=candidate.lora_rank)
    return value


def _routing_observation(status: dict) -> dict:
    value = {name: status[name] for name in (
        "target", "training_backend", "gpu_accelerated", "cpu_count",
        "memory_available_mb", "active_jobs") if name in status}
    gpu = status.get("gpu")
    if isinstance(gpu, dict) and gpu.get("detected"):
        value["gpu"] = {name: gpu[name] for name in (
            "name", "memory_free_mb", "utilization_percent", "expected_match") if name in gpu}
    return value


def validate_proposal(proposal: dict, config: TaskConfig, snapshot_sha: str, seen: set[str] | None = None,
                      available_targets: set[str] | None = None) -> list[Candidate]:
    required = {"snapshot_sha", "candidates", "training_target", "routing_reason"}
    if (not isinstance(proposal, dict) or not required.issubset(proposal)
            or set(proposal) - required - {"experiment_specs"} or proposal["snapshot_sha"] != snapshot_sha):
        raise GateError("PROPOSAL_EVIDENCE_OR_SCHEMA")
    if proposal["training_target"] not in config.execution.allowed_targets:
        raise GateError("TRAINING_TARGET_NOT_ALLOWED")
    if available_targets is not None and proposal["training_target"] not in available_targets:
        raise GateError("TRAINING_TARGET_UNAVAILABLE")
    reason = proposal["routing_reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 240 or any(ord(c) < 32 for c in reason):
        raise GateError("ROUTING_REASON")
    raw = proposal["candidates"]
    if not isinstance(raw, list) or not raw or len(raw) > config.budget.max_candidates:
        raise GateError("CANDIDATE_BUDGET")
    candidates = [Candidate.from_dict(x) for x in raw]
    for candidate in candidates:
        candidate.validate(config)
    if "experiment_specs" in proposal:
        from .experiments import compile_experiment
        for value in proposal["experiment_specs"]:
            candidate, target, _ = compile_experiment(
                value, config, evidence_refs={snapshot_sha[:24]},
                available_targets=available_targets or set(config.execution.allowed_targets))
            if candidate not in candidates or target != proposal["training_target"]:
                raise GateError("EXPERIMENT_EXECUTION_MISMATCH")
    fingerprints = [c.candidate_id for c in candidates]
    if len(set(fingerprints)) != len(fingerprints):
        raise GateError("DUPLICATE_CANDIDATE")
    fingerprint = proposal_fingerprint(proposal)
    if seen is not None and (fingerprint in seen or digest(sorted(fingerprints)) in seen):
        raise GateError("NO_PROGRESS")
    return candidates


def make_proposal(config: TaskConfig, snapshot_sha: str, summary: dict, *, transport=request_chat_bytes,
                  routing_transport=None, execution_status: list[dict] | None = None,
                  repair_context: dict | None = None, token_limit: int | None = None,
                  trial_history=(), trace=None, on_request=None, cancelled=None) -> tuple[dict, int, list[dict]]:
    candidates = (allowed_candidates(config)[:config.budget.max_candidates] if config.planner.kind == "rule"
                  else candidate_pool(config))
    status = execution_status if execution_status is not None else probe_training_targets(config, transport=routing_transport)
    available = {item["target"] for item in status if item.get("available")}
    if not available:
        raise GateError("NO_TRAINING_TARGET_AVAILABLE")
    if config.planner.kind == "rule":
        if config.execution.default_target in available:
            target = config.execution.default_target
            reason = "trusted rule default"
        else:
            target = sorted(available)[0]
            reason = f"trusted rule fallback: {config.execution.default_target} unavailable; selected {target}"
        proposal = {"snapshot_sha": snapshot_sha, "candidates": [asdict(c) for c in candidates],
                    "training_target": target, "routing_reason": reason}
        validate_proposal(proposal, config, snapshot_sha, available_targets=available)
        return proposal, 0, status
    if not config.planner.allow_external:
        raise GateError("EXTERNAL_PLANNER_NOT_AUTHORIZED")
    url = os.environ.get(config.planner.endpoint_env, "")
    key = os.environ.get(config.planner.key_env, "")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or not key:
        raise GateError("PLANNER_CONFIGURATION")
    # The full snapshot identity (which includes Test) stays in the control plane.
    # A request-scoped alias does not reveal even changes in Test-only values.
    evidence_ref = "s0"
    offered_refs = {f"c{index}": candidate for index, candidate in enumerate(candidates)}
    initial_limit = min(config.planner.initial_candidates, max(1, config.budget.max_candidates -
                        (1 if config.budget.max_repairs and config.budget.max_candidates > 1 else 0)))
    observation = {**build_context(config, profile=summary, history=trial_history),
                   "rows": summary["rows"], "variable_count": len(config.variables), "missing_fraction": summary["missing_fraction"],
                   "evidence_ref": evidence_ref,
                   "execution_budget": {"initial_candidates": initial_limit, "total_candidates": config.budget.max_candidates},
                   "allowed_candidates": [_candidate_observation(candidate, candidate_ref)
                                          for candidate_ref, candidate in offered_refs.items()],
                   "available_training_targets": [_routing_observation(item) for item in status if item.get("available")]}
    observation["experiment_capabilities"] = _capability_observation(config)
    observation["experiment_schema"] = {
        "objective": "short goal", "hypothesis": "testable hypothesis",
        "candidate": {"family": "ridge", "input_steps": config.search.inputs[0],
                      "history_steps": config.search.histories[0], "alpha": config.search.ridge_alphas[0],
                      "features": []}, "contract_ref": "contract",
        "evaluation_protocol": "rolling-origin-v1", "evidence_refs": [evidence_ref],
        "resources": {"training_target": sorted(available)[0],
                      "seconds": config.budget.max_candidate_seconds,
                      "memory_mb": config.budget.max_memory_mb}}
    if config.search.feature_operations and config.planner.example_limit:
        observation["examples"] = [{"hypothesis": "Test a causal summary of recent variability.",
                                    "feature": {"operation": config.search.feature_operations[0],
                                                "source_field": "target:0", "window": 2}}]
    if catalog_only(config):
        observation.pop("experiment_capabilities")
        observation.pop("experiment_schema")
        observation["available_training_targets"] = [
            {"target": item["target"], "training_backend": item.get("training_backend")}
            for item in status if item.get("available")]
    if repair_context:
        issue = repair_context.get("last_gate_issue") or {}
        observation["repair_context"] = {"round": repair_context.get("round"), "code": issue.get("code"),
                                         "required_action": "Correct the schema or known parameter-direction contradiction."}
    available_tokens = config.budget.max_llm_tokens if token_limit is None else token_limit
    if isinstance(available_tokens, bool) or not isinstance(available_tokens, int) or available_tokens < 1:
        raise GateError("PLANNER_TOKEN_BUDGET")
    prompt = files("online_ts").joinpath("prompts/" + planner_prompt_version(config).replace("-", "_") + ".txt").read_text(encoding="utf-8")
    output_limit = config.planner.short_output_tokens if catalog_only(config) else config.planner.spec_output_tokens
    messages, prompt_reserve, output_limit, omitted = fit_messages(prompt, observation, available_tokens, output_limit)
    body = {"model": config.planner.model, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "seed": config.seed, "max_tokens": output_limit,
            "reasoning_effort": "none",
            "messages": messages}
    request = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=canonical(body).encode(), method="POST", headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    trace = trace or LLMCallTrace("planning-" + digest(body)[:24], "planning")
    trace.prepare(request, config.planner.timeout_seconds, prompt_reserve, output_limit, omitted)
    trace.data["prompt_version"] = planner_prompt_version(config)
    trace.data["candidate_pool"] = {ref: asdict(candidate) for ref, candidate in offered_refs.items()}
    tokens = None
    response = None
    try:
        response, decision = _read_response(transport, request, config, trace, on_request, cancelled)
        tokens = response["usage"]["total_tokens"]
        if not isinstance(tokens, int) or isinstance(tokens, bool) or not 0 < tokens <= available_tokens:
            raise GateError("PLANNER_TOKEN_BUDGET")
        validation_start = time.monotonic()
        if catalog_only(config) and isinstance(decision, dict) and "experiments" in decision:
            raise GateError("PROPOSAL_EVIDENCE_OR_SCHEMA")
        if (not isinstance(decision, dict)
                or set(decision) not in ({"evidence_ref", "candidate_refs", "training_target", "routing_reason"},
                                         {"evidence_ref", "candidate_refs", "experiments", "training_target", "routing_reason"})
                or decision["evidence_ref"] != evidence_ref):
            raise GateError("PROPOSAL_EVIDENCE_OR_SCHEMA")
        candidate_refs = decision["candidate_refs"]
        if not isinstance(candidate_refs, list) or len(candidate_refs) > config.budget.max_candidates:
            raise GateError("CANDIDATE_BUDGET")
        if len(candidate_refs) != len(set(candidate_refs)):
            raise GateError("DUPLICATE_CANDIDATE")
        if any(not isinstance(candidate_ref, str) or candidate_ref not in offered_refs for candidate_ref in candidate_refs):
            raise GateError("CANDIDATE_NOT_OFFERED")
        proposal = {"snapshot_sha": snapshot_sha,
                    "candidates": [asdict(offered_refs[candidate_ref]) for candidate_ref in candidate_refs],
                    "training_target": decision["training_target"],
                    "routing_reason": decision["routing_reason"]}
        if "experiments" in decision:
            from .experiments import compile_experiment
            specs = decision["experiments"]
            if not isinstance(specs, list) or len(specs) > config.budget.max_candidates:
                raise GateError("CANDIDATE_BUDGET")
            expanded = []
            for value in specs:
                value = _expand_spec(value, config, {evidence_ref: snapshot_sha[:24]})
                candidate, target, _ = compile_experiment(
                    value, config, evidence_refs={snapshot_sha[:24]}, available_targets=available)
                if target != decision["training_target"]:
                    raise GateError("EXPERIMENT_EXECUTION_MISMATCH")
                proposal["candidates"].append(asdict(candidate))
                expanded.append(value)
            proposal["experiment_specs"] = expanded
            # This ordering is part of the issued protocol and fingerprint: test
            # new hypotheses first, then their selected catalogue comparisons.
            proposal["candidates"] = (proposal["candidates"][len(candidate_refs):]
                                      + proposal["candidates"][:len(candidate_refs)])
        if len(proposal["candidates"]) > initial_limit:
            raise GateError("INITIAL_CANDIDATE_LIMIT")
        validate_proposal(proposal, config, snapshot_sha, available_targets=available)
        trace.data["validation_duration_ms"] = round((time.monotonic() - validation_start) * 1000, 3)
        trace.mark("legal_proposal_ms")
        trace.finish()
    except PlannerResponseError:
        raise
    except GateError as error:
        tokens = tokens if type(tokens) is int else (trace.data.get("usage") or {}).get("total_tokens")
        trace.finish(str(error))
        billed = _billed_response_error(str(error), tokens, response or trace.data.get("model_response"))
        if billed is not None:
            raise billed from None
        raise
    except Exception:
        tokens = tokens if type(tokens) is int else (trace.data.get("usage") or {}).get("total_tokens")
        trace.finish("PROPOSAL_EVIDENCE_OR_SCHEMA")
        billed = _billed_response_error("PROPOSAL_EVIDENCE_OR_SCHEMA", tokens, response or trace.data.get("model_response"))
        if billed is not None:
            raise billed from None
        raise GateError("PLANNER_FAILED") from None
    return proposal, tokens, status


def make_feedback_proposal(config: TaskConfig, context_value: dict, *, transport=request_chat_bytes,
                           routing_transport=None, execution_status: list[dict] | None = None,
                           token_limit: int | None = None, trace=None, on_request=None,
                           cancelled=None) -> tuple[dict, int, list[dict]]:
    """Call the single authorized Agent interface with a bounded, sanitized context."""
    context = AgentContext.from_dict(context_value)
    status = execution_status if execution_status is not None else probe_training_targets(
        config, transport=routing_transport)
    available = {item["target"] for item in status if item.get("available")}
    allowed_targets = set(context.allowed_changes.get("training_targets", ())) & available
    if not allowed_targets:
        raise GateError("NO_TRAINING_TARGET_AVAILABLE")
    remaining = context.remaining_budget
    if (remaining.get("repair_rounds", 0) < 0 or remaining.get("llm_tokens", 0) < 1
            or remaining.get("candidate_slots", 0) < 1):
        raise GateError("FEEDBACK_BUDGET_EXHAUSTED")
    if config.planner.kind != "external" or not config.planner.allow_external:
        raise GateError("EXTERNAL_PLANNER_NOT_AUTHORIZED")
    url = os.environ.get(config.planner.endpoint_env, "")
    key = os.environ.get(config.planner.key_env, "")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or not key:
        raise GateError("PLANNER_CONFIGURATION")
    available_tokens = min(config.budget.max_llm_tokens, remaining["llm_tokens"])
    if token_limit is not None:
        available_tokens = min(available_tokens, token_limit)
    if isinstance(available_tokens, bool) or not isinstance(available_tokens, int) or available_tokens < 1:
        raise GateError("PLANNER_TOKEN_BUDGET")
    observation, issue_aliases, evidence_aliases = catalog_feedback_observation(context)
    observation.update(build_context(config, profile=context.current_plan.get("task_profile"),
                                     history=context.current_plan.get("trial_history", ()),
                                     issues=context.gate_issues, candidate=context.outcome["effective_config"]))
    from .feedback import _compact_candidate
    observation["current_config"] = _compact_candidate(Candidate.from_dict(context.outcome["effective_config"])) if context.outcome["effective_config"] else {}
    observation["program_facts"] = {"metric_deltas": context.comparisons.get("metric_deltas", {}),
                                     "best_score_delta": context.comparisons.get("best_score_delta"),
                                     "quality_improved": context.comparisons.get("quality_improved")}
    parent_aliases = {f"p{i}": c["candidate_id"] for i, c in enumerate(context.current_plan.get("executed_candidates", []))}
    parent_candidates = {f"p{i}": c["candidate"] for i, c in enumerate(context.current_plan.get("executed_candidates", []))}
    prompt = feedback_prompt()
    prompt_version = FEEDBACK_PROMPT_VERSION
    if catalog_only(config):
        prompt_version = "catalog-feedback-v2"
        prompt = files("online_ts").joinpath("prompts/catalog_feedback_v2.txt").read_text(encoding="utf-8")
    else:
        observation["experiment_capabilities"] = _capability_observation(config)
        observation["spec_bindings"] = {"contract_ref": "contract", "feedback_id": "feedback", "parent_candidates": parent_candidates}
        observation["experiment_schema"] = {
            "objective": "short goal", "hypothesis": "testable hypothesis", "candidate": "complete candidate object",
            "contract_ref": "contract", "evaluation_protocol": "rolling-origin-v1",
            "feedback_id": "feedback", "parent_candidate_id": next(iter(parent_aliases), None),
            "evidence_refs": list(evidence_aliases)[:1],
            "resources": {"training_target": sorted(allowed_targets)[0],
                          "seconds": config.budget.max_candidate_seconds, "memory_mb": config.budget.max_memory_mb}}
    messages, prompt_reserve, output_limit, omitted = fit_messages(
        prompt, observation, available_tokens, config.planner.feedback_output_tokens)
    body = {"model": config.planner.model, "temperature": 0.2, "top_p": 0.8,
            "seed": config.seed, "max_tokens": output_limit,
            "reasoning_effort": "none",
            "messages": messages}
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=canonical(body).encode(), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    trace = trace or LLMCallTrace("feedback-" + digest(body)[:24], "feedback")
    trace.prepare(request, config.planner.timeout_seconds, prompt_reserve, output_limit, omitted)
    trace.data.update(prompt_version=prompt_version, issue_aliases=issue_aliases,
                      evidence_aliases=evidence_aliases, parent_aliases=parent_aliases)
    tokens = None
    response = None
    try:
        response, decision = _read_response(transport, request, config, trace, on_request, cancelled)
        tokens = response["usage"]["total_tokens"]
        if not isinstance(tokens, int) or isinstance(tokens, bool) or not 0 < tokens <= available_tokens:
            raise GateError("PLANNER_TOKEN_BUDGET")
        validation_start = time.monotonic()
        expected = {"action", "issue_ids", "diagnosis_summary", "evidence_refs", "hypothesis",
                    "proposed_changes", "expected_direction", "risks", "focus_metrics"}
        if not isinstance(decision, dict) or set(decision) != expected:
            raise GateError("FEEDBACK_PROPOSAL_SCHEMA")
        if issue_aliases is not None:
            if (not isinstance(decision["issue_ids"], list) or not isinstance(decision["evidence_refs"], list)
                    or any(ref not in issue_aliases for ref in decision["issue_ids"])
                    or any(ref not in evidence_aliases for ref in decision["evidence_refs"])):
                raise GateError("FEEDBACK_PROPOSAL_EVIDENCE")
            fields = set(decision["proposed_changes"])
            if decision["action"] == "experiment_patch" and (fields not in ({"candidate_ref", "training_target"}, {"experiment_spec", "training_target"})
                    or (catalog_only(config) and "experiment_spec" in fields)):
                raise GateError("FEEDBACK_CHANGE_NOT_ALLOWED")
            if "experiment_spec" in fields:
                decision["proposed_changes"]["experiment_spec"] = _expand_spec(
                    decision["proposed_changes"]["experiment_spec"], config, evidence_aliases,
                    feedback_id=context.feedback_id, parent_aliases=parent_aliases)
            decision["issue_ids"] = [issue_aliases[ref] for ref in decision["issue_ids"]]
            decision["evidence_refs"] = [evidence_aliases[ref] for ref in decision["evidence_refs"]]
        created_at = time.time_ns()
        identity = {"run_id": context.run_id, "feedback_id": context.feedback_id,
                    "base_plan_version": context.plan_version, "decision": decision}
        proposal = FeedbackProposal(
            FEEDBACK_SCHEMA_VERSION, "proposal-" + digest(identity)[:24], context.run_id,
            decision["action"], context.plan_id, context.plan_version, context.feedback_id,
            tuple(decision["issue_ids"]), decision["diagnosis_summary"],
            tuple(decision["evidence_refs"]), decision["hypothesis"],
            decision["proposed_changes"], decision["expected_direction"],
            tuple(decision["risks"]), tuple(decision["focus_metrics"]), created_at,
            prompt_version, config.planner.model,
        ).to_dict()
        # Complete deterministic admission before calling the response legal. The
        # Worker rechecks its current plan and durable replay set before committing.
        if decision["action"] == "experiment_patch":
            from .experiments import compile_experiment
            from .context import validate_parameter_language
            if "experiment_spec" in decision["proposed_changes"]:
                spec = decision["proposed_changes"]["experiment_spec"]
                candidate, _, _ = compile_experiment(spec, config, evidence_refs=context.evidence_refs,
                    available_targets=allowed_targets, parent_ids=parent_aliases.values(), feedback_id=context.feedback_id)
                before = next((c["candidate"] for c in context.current_plan.get("executed_candidates", [])
                               if c["candidate_id"] == spec.get("parent_candidate_id")), context.outcome["effective_config"])
            else:
                offered = {item["candidate_ref"]: item["candidate"] for item in context.allowed_changes["candidate_catalog"]}
                ref = decision["proposed_changes"]["candidate_ref"]
                if ref not in offered:
                    raise GateError("CANDIDATE_NOT_OFFERED")
                candidate, before = Candidate.from_dict(offered[ref]), context.outcome["effective_config"]
            trace.data["parameter_facts"] = validate_parameter_language(
                (decision["hypothesis"] or "") + " " + decision["diagnosis_summary"], before, asdict(candidate))
            if candidate.candidate_id in context.current_plan.get("candidate_ids", []):
                raise GateError("NO_PROGRESS")
            if decision["proposed_changes"]["training_target"] not in allowed_targets:
                raise GateError("TRAINING_TARGET_NOT_ALLOWED")
        trace.data["validation_duration_ms"] = round((time.monotonic() - validation_start) * 1000, 3)
        trace.mark("legal_proposal_ms")
        trace.finish()
    except PlannerResponseError:
        raise
    except GateError as error:
        tokens = tokens if type(tokens) is int else (trace.data.get("usage") or {}).get("total_tokens")
        trace.finish(str(error))
        billed = _billed_response_error(str(error), tokens, response or trace.data.get("model_response"))
        if billed is not None:
            raise billed from None
        raise
    except Exception:
        tokens = tokens if type(tokens) is int else (trace.data.get("usage") or {}).get("total_tokens")
        trace.finish("FEEDBACK_PROPOSAL_SCHEMA")
        billed = _billed_response_error("FEEDBACK_PROPOSAL_SCHEMA", tokens, response or trace.data.get("model_response"))
        if billed is not None:
            raise billed from None
        raise GateError("PLANNER_FAILED") from None
    return proposal, tokens, status
