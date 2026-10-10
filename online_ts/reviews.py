"""Append-only, deterministic candidate cards and evidence-bounded explanations."""
from __future__ import annotations

import json
import math
import re
import time

from .config import GateError, canonical, digest


REVIEW_PROMPT_VERSION = "model-reviewer-v1"


def numeric_delta(current, reference):
    if any(type(v) not in {int, float} or not math.isfinite(v) for v in (current, reference)):
        return {"absolute": None, "relative_percent": None, "denominator": reference}
    return {"absolute": current - reference,
            "relative_percent": None if reference == 0 else 100 * (current - reference) / abs(reference),
            "denominator": reference, "direction": "right_minus_left_not_a_quality_verdict"}


def comparison(left: dict, right: dict) -> dict:
    identity = ("snapshot_sha", "evaluation_protocol", "sample_fingerprint")
    comparable = (all(left.get(k) is not None and left.get(k) == right.get(k) for k in identity)
                  and left["stage"] == right["stage"])
    deltas = {}
    if comparable:
        for name in set(left["facts"]) & set(right["facts"]):
            a, b = left["facts"][name], right["facts"][name]
            if type(a) in {int, float} and type(b) in {int, float}:
                deltas[name] = numeric_delta(b, a)
    return {"left": left["review_id"], "right": right["review_id"],
            "comparable": comparable, "reason": None if comparable else "SAMPLE_OR_PROTOCOL_MISMATCH",
            "deltas_right_minus_left": deltas,
            "right_gate": right["program_state"]["gate_passed"],
            "recommendation": "仅比较同口径且通过 Gate 的候选；模型评价不构成发布批准。"}


def _flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for name, item in value.items():
            result.update(_flatten(item, f"{prefix}.{name}" if prefix else name))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}.{i}"))
    elif value is None or type(value) in {int, float, bool, str}:
        result[prefix] = value
    return result


def build_card(job, candidate, stage, evidence, *, model_release_id=None, deployment="not_deployed"):
    from .models import Candidate
    c = Candidate.from_dict(candidate)
    if stage == "development_validation":
        gate = bool(evidence.get("eligible"))
        metrics = {k: evidence.get(k) for k in ("score", "score_std", "folds")}
        state = "evaluated" if "score" in evidence else "failed"
        protocol = "rolling-origin-v1"
        samples = [{k: f.get(k) for k in ("issue_times", "scale", "validation_start", "validation_end")}
                   for f in evidence.get("folds", [])]
    elif stage == "final_test":
        gate = bool(evidence.get("passed"))
        metrics = {k: evidence.get(k) for k in ("offline", "baselines", "champion_comparison", "probabilistic", "label_coverage")}
        state, protocol = "evaluated", "sealed-final-test-v1"
        samples = {k: evidence.get(k) for k in ("test_start", "test_end", "label_coverage")}
    elif stage == "online_mature":
        gate = None  # monitoring observations alone do not grant release eligibility
        metrics = evidence
        state, protocol = "evaluated", "online-mature-v1"
        samples = {k: evidence.get(k) for k in ("window_start", "window_end", "label_fingerprint")}
    else:
        gate, metrics = None, {}
        state, protocol, samples = evidence.get("status", "failed"), "execution-v1", None
    facts = _flatten(metrics, "metrics")
    facts.update(_flatten(evidence.get("resource_observation", {}), "resources"))
    issues = evidence.get("issues", [evidence["code"]] if evidence.get("code") else [])
    summary = ("开发验证完成；尚不构成最终准入或上线批准。" if stage == "development_validation" and gate
               else "最终准入通过；发布仍受在线观察及授权约束。" if stage == "final_test" and gate
               else "当前 Gate 拒绝；保留现有模型。" if gate is False
               else "在线标签证据已更新；请查看成熟度与覆盖。" if stage == "online_mature"
               else "训练完成，评价待完成。" if state == "trained"
               else "候选已取消。" if state == "cancelled"
               else "候选未完成训练，暂无质量结论。")
    evidence_version = digest(evidence)
    identity = {"run": job["run_id"], "candidate": c.candidate_id, "evidence": evidence_version,
                "stage": stage, "release": model_release_id, "deployment": deployment,
                "prompt": REVIEW_PROMPT_VERSION}
    specs = job["plan"].get("experiment_history", []) + job["plan"].get("experiment_specs", [])
    parent = next((s.get("parent_candidate_id") for s in specs
                   if Candidate.from_dict(s["candidate"]).candidate_id == c.candidate_id), None)
    return {"schema_version": 1, "review_id": "review-" + digest(identity)[:24],
            "run_id": job["run_id"], "candidate_id": c.candidate_id,
            "model_release_id": model_release_id, "parent_candidate_id": parent,
            "plan_version": job["checkpoint"]["plan_version"],
            "policy_version": evidence.get("config_version"),
            "candidate": candidate, "feature_version": digest(c.features),
            "adaptation_mode": c.adaptation_mode,
            "weights_updated": ((c.family == "ridge" or c.adaptation_mode == "lora")
                                if state in {"evaluated", "trained"} else None),
            "snapshot_sha": job["snapshot"].get("sha256"), "stage": stage,
            "evaluation_protocol": protocol,
            "sample_fingerprint": digest(samples) if samples else None,
            "evidence_version": evidence_version, "evidence_refs": ["evidence:" + evidence_version],
            "source_evidence": evidence,
            "attempt_ids": [o["attempt_id"] for o in job["checkpoint"].get("outcomes", [])
                            if o.get("candidate_id") == c.candidate_id and o.get("evidence_scope") == stage],
            "created_at": time.time_ns(), "prompt_version": REVIEW_PROMPT_VERSION,
            "summary": summary, "program_state": {"training_status": state, "gate_passed": gate,
                                                     "deployment": deployment, "issues": issues},
            "facts": facts, "metrics": metrics,
            "limitations": ["未验证工厂现场适用性。", "未测量的资源、设备分组和业务收益不填零。",
                            "零样本未在项目数据上更新权重。" if c.adaptation_mode == "zero_shot"
                            else "相近训练/验证误差不能证明不存在过拟合。"],
            "llm_status": "pending", "llm_explanation": None,
            "actions": ["evidence", "compare", "ask", "draft_experiment"]}


def insert_card(db, task_key, card):
    db.execute("INSERT OR IGNORE INTO model_reviews VALUES(?,?,?,?,?,?,?,?,?)",
               (card["review_id"], task_key, card["run_id"], card["candidate_id"], card["stage"],
                card["evidence_version"], card["prompt_version"], card["created_at"], canonical(card)))


def persist_job_cards(db, job, checkpoint, result, status):
    job = {**job, "checkpoint": checkpoint}
    for report in checkpoint.get("reports", []):
        stage = "development_validation" if "score" in report else "execution"
        insert_card(db, job["task_key"], build_card(job, report["candidate"], stage, report))
    for outcome in checkpoint.get("outcomes", []):
        if outcome.get("candidate_id") and outcome["evidence_scope"] in {"execution", "resource"}:
            insert_card(db, job["task_key"], build_card(job, outcome["effective_config"], "execution",
                        {"status": "failed", "issues": outcome["issue_codes"], "attempt_id": outcome["attempt_id"]}))
    if result and "candidate" in result and "offline" in result:
        insert_card(db, job["task_key"], build_card(job, result["candidate"], "final_test", result))
    if status in {"FAILED", "CANCELLED", "COMPLETED"}:
        from .models import Candidate
        reported = {Candidate.from_dict(r["candidate"]).candidate_id for r in checkpoint.get("reports", [])}
        for candidate in job["plan"].get("candidates", []):
            if Candidate.from_dict(candidate).candidate_id not in reported:
                insert_card(db, job["task_key"], build_card(job, candidate, "execution",
                            {"status": "cancelled" if status == "CANCELLED" else "not_completed",
                             "issues": [checkpoint.get("terminal_reason") or status]}))


def list_cards(store, task_key, *, candidate_id=None, history=False, limit=100):
    with store.connection() as db:
        rows = db.execute("SELECT payload FROM model_reviews WHERE task_key=? ORDER BY created_ns,review_id",
                          (task_key,)).fetchall()
        versions = _explanation_versions(db, task_key)
    cards = [_with_explanation(json.loads(row[0]), versions) for row in rows]
    if candidate_id:
        cards = [c for c in cards if c["candidate_id"] == candidate_id]
    latest = {}
    for c in cards:
        latest[(c["run_id"], c["candidate_id"], c["stage"])] = c["review_id"]
    for c in cards:
        c["current_for_stage"] = latest[(c["run_id"], c["candidate_id"], c["stage"])] == c["review_id"]
    if not history:
        cards = [c for c in cards if c["current_for_stage"]]
    return cards[-min(max(limit, 1), 500):]


def get_card(store, task_key, review_id):
    with store.connection() as db:
        row = db.execute("SELECT payload FROM model_reviews WHERE task_key=? AND review_id=?",
                         (task_key, review_id)).fetchone()
        if row is None:
            raise GateError("REVIEW_NOT_AUTHORIZED_OR_NOT_FOUND")
        card = json.loads(row[0])
        latest = db.execute("SELECT review_id FROM model_reviews WHERE task_key=? AND run_id=? AND candidate_id=? AND stage=? ORDER BY created_ns DESC,review_id DESC LIMIT 1",
                            (task_key, card["run_id"], card["candidate_id"], card["stage"])).fetchone()[0]
        card["current_for_stage"] = latest == review_id
        card = _with_explanation(card, _explanation_versions(db, task_key, review_id))
    if digest(card["source_evidence"]) != card["evidence_version"]:
        raise GateError("REVIEW_EVIDENCE_TAMPERED")
    return card


def _explanation_versions(db, task_key, review_id=None):
    rows = db.execute("""SELECT kind,payload FROM feedback_messages WHERE task_key=?
        AND kind IN ('model_explanation','review_request')
        AND (? IS NULL OR json_extract(payload,'$.review_id')=?) ORDER BY created_ns,message_id""",
        (task_key, review_id, review_id)).fetchall()
    versions = {}
    for row in rows:
        payload = json.loads(row["payload"])
        versions.setdefault(payload["review_id"], []).append((row["kind"], payload))
    return versions


def _with_explanation(card, versions):
    history = versions.get(card["review_id"], [])
    explanations = [payload for kind, payload in history if kind == "model_explanation"]
    if explanations:
        last = explanations[-1]
        if last["evidence_version"] != card["evidence_version"]:
            raise GateError("EXPLANATION_EVIDENCE_MISMATCH")
        card.update(llm_status=last["status"], llm_explanation=last["explanation"],
                    explanation_revision=last["message_id"],
                    explanation_history=[v["message_id"] for v in explanations])
    elif any(kind == "review_request" for kind, _ in history):
        card["llm_status"] = "queued"
    return card


def append_explanation(store, task_key, card, state, explanation, model):
    value = {"review_id": card["review_id"], "evidence_version": card["evidence_version"],
             "prompt_version": REVIEW_PROMPT_VERSION, "model": model,
             "status": state, "explanation": explanation}
    message_id = "explanation-" + digest(value)
    value["message_id"] = message_id
    with store.transaction() as db:
        db.execute("INSERT OR IGNORE INTO feedback_messages VALUES(?,?,?,?,?,?,?)",
                   (message_id, task_key, card["run_id"], "model_explanation", card["plan_version"],
                    time.time_ns(), canonical(value)))
        store._event(db, task_key, "MODEL_EXPLANATION", {"review_id": card["review_id"], "status": state,
                                                       "revision": message_id}, time.time_ns(), message_id)


def queue_explanation(store, task_key, review_id):
    card = get_card(store, task_key, review_id)
    if card["stage"] == "final_test":
        append_explanation(store, task_key, card, "unavailable:FINAL_TEST_LLM_EXCLUDED", None, None)
        return get_card(store, task_key, review_id)
    if card["llm_status"] != "pending":
        return card
    identity = "review-request-" + digest({"review_id": review_id, "prompt": REVIEW_PROMPT_VERSION})
    value = {"review_id": review_id, "evidence_version": card["evidence_version"], "request_id": identity}
    with store.transaction() as db:
        db.execute("INSERT OR IGNORE INTO feedback_messages VALUES(?,?,?,?,?,?,?)",
                   (identity, task_key, card["run_id"], "review_request", card["plan_version"], time.time_ns(), canonical(value)))
    return get_card(store, task_key, review_id)


def summary_facts(card, limit=16):
    facts = card["facts"]
    priorities = ("metrics.score", "metrics.score_std", "metrics.offline.normalized_mae",
                  "metrics.offline.worst_variable_nmae", "metrics.offline.worst_horizon_nmae", "metrics.label_coverage")
    result = {k: facts[k] for k in priorities if k in facts}
    for key in facts:
        if len(result) >= limit:
            break
        value = facts[key]
        if type(value) in {int, float, bool} or value is None:
            result[key] = value
    return result


def validate_explanation(value, card):
    fields = {"evidence_based_observations", "hypotheses", "limitations", "advisory_recommendations"}
    if not isinstance(value, dict) or set(value) != fields:
        raise GateError("REVIEW_EXPLANATION_SCHEMA")
    observations = value["evidence_based_observations"]
    if not isinstance(observations, list) or len(observations) > 8:
        raise GateError("REVIEW_OBSERVATIONS")
    for observation in observations:
        if not isinstance(observation, dict) or set(observation) != {"evidence_ref", "field", "value"}:
            raise GateError("REVIEW_OBSERVATION_SCHEMA")
        field = observation["field"]
        expected = card["facts"].get(field)
        actual = observation["value"]
        numeric_match = type(expected) in {int, float} and type(actual) in {int, float} and expected == actual
        if (observation["evidence_ref"] not in card["evidence_refs"] or field not in card["facts"]
                or (not numeric_match and canonical(actual) != canonical(expected))):
            raise GateError("REVIEW_FACT_MISMATCH")
    for name in fields - {"evidence_based_observations"}:
        if not isinstance(value[name], list) or len(value[name]) > 6:
            raise GateError("REVIEW_ADVISORY_SCHEMA")
        for text in value[name]:
            if not isinstance(text, str) or len(text) > 400 or not text.strip():
                raise GateError("REVIEW_ADVISORY_SCHEMA")
            # All numerical statements are rendered from exact field references instead.
            if re.search(r"\d|%|已上线|现场已验证|已经生产可用|忽略.*[Gg]ate|ignore.*gate", text):
                raise GateError("REVIEW_UNGROUNDED_CLAIM")
    return value


def render_card(card, *, details=False):
    status = card["program_state"]
    lines = [f"候选 {card['candidate_id']} | {card['candidate']['family']} | {card['stage']}",
             card["summary"], f"报告 {card['review_id']}",
             f"训练：{status['training_status']}；Gate：{status['gate_passed']}；部署：{status['deployment']}",
             f"适配：{card['adaptation_mode']}；证据版本：{card['evidence_version']}",
             "LLM解释：" + card["llm_status"]]
    if card.get("current_for_stage") is False:
        lines.insert(0, "历史评价版本：已有更新证据，本卡仅适用于记录时的阶段。")
    names = [n for n in card["facts"] if n in {"metrics.score", "metrics.score_std", "metrics.offline.normalized_mae"}]
    for name in names:
        lines.append(f"{name}: {card['facts'][name]}")
    if status["issues"]:
        lines.append("程序问题：" + ", ".join(status["issues"]))
    lines.extend(card["limitations"])
    explanation = card.get("llm_explanation")
    if explanation:
        for name, title in (("hypotheses", "待验证假设"), ("limitations", "局限"),
                            ("advisory_recommendations", "建议")):
            lines.extend(title + "：" + t for t in explanation[name])
    if details:
        lines.append(json.dumps(card["metrics"], ensure_ascii=False, indent=2))
        lines.append("证据引用：" + ", ".join(card["evidence_refs"]))
    return "\n".join(lines)
