"""One restricted LLM transport with node-specific prompts and durable reservations."""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from importlib.resources import files

from .config import GateError, canonical, digest
from .llm_transport import request_chat_bytes


ROLE_PROMPTS = {"review": "model_reviewer_v1", "question": "human_copilot_v1",
                "intent": "human_copilot_v1", "data_analyst": "data_analyst_v1",
                "feature_designer": "feature_designer_v1", "release_reporter": "release_reporter_v1"}


def call_agent(store, config, task_key, run_id, role, evidence, *, operation_id, transport=request_chat_bytes):
    owner = uuid.uuid4().hex
    if not store.acquire_resource("llm-call", owner, task_key, run_id, role, config.planner.timeout_seconds + 30):
        raise GateError("LLM_BUSY")
    try:
        if transport is request_chat_bytes:
            def bounded(request, timeout):
                return request_chat_bytes(request, timeout, cancelled=lambda: store.task(task_key)[1]["stop_requested"])
            transport = bounded
        return _call_agent(store, config, task_key, run_id, role, evidence, operation_id=operation_id, transport=transport)
    finally:
        store.release_resource("llm-call", owner)


def _call_agent(store, config, task_key, run_id, role, evidence, *, operation_id, transport):
    if role not in ROLE_PROMPTS:
        raise GateError("AGENT_ROLE")
    if config.planner.kind != "external" or not config.planner.allow_external:
        raise GateError("EXTERNAL_PLANNER_NOT_AUTHORIZED")
    # The caller supplies ONLY a constructed summary. This API has no tools or filesystem access.
    prompt_version = ROLE_PROMPTS[role]
    messages = [{"role": "system", "content": files("online_ts").joinpath(
                    "prompts/" + prompt_version + ".txt").read_text(encoding="utf-8")},
                {"role": "user", "content": canonical(evidence)}]
    identity = digest({"messages": messages, "model": config.planner.model})
    category = role if role in {"review", "question", "intent"} else "question"
    budget = getattr(config.collaboration, category + "_tokens")
    limit = min(config.collaboration.max_call_tokens, budget)
    # UTF-8 bytes are a conservative upper bound for byte-level tokenization.
    prompt_reserve = len(canonical(messages).encode("utf-8")) + 128
    if prompt_reserve + 256 > limit:
        raise GateError("AGENT_PROMPT_BUDGET")
    url, key = os.environ.get(config.planner.endpoint_env, ""), os.environ.get(config.planner.key_env, "")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or not key:
        raise GateError("PLANNER_CONFIGURATION")
    with store.transaction() as db:
        old = db.execute("SELECT * FROM agent_calls WHERE operation_id=?", (operation_id,)).fetchone()
        if old:
            payload = json.loads(old["payload"])
            if old["task_key"] != task_key or old["run_id"] != run_id or payload["identity"] != identity:
                raise GateError("AGENT_OPERATION_CONFLICT")
            if old["status"] == "completed":
                return payload["response"]
            raise GateError("AGENT_CALL_UNCERTAIN" if old["status"] == "reserved" else "AGENT_CALL_FAILED")
        total, calls = db.execute("SELECT COALESCE(SUM(COALESCE(actual_tokens,reserved_tokens)),0),COUNT(*) FROM agent_calls WHERE task_key=? AND run_id=? AND role=?",
                                 (task_key, run_id, category)).fetchone()
        if total + limit > budget or calls >= config.collaboration.max_calls_per_role:
            raise GateError("AGENT_ROLE_BUDGET")
        payload = {"identity": identity, "prompt_version": prompt_version,
                   "model": config.planner.model, "request": evidence}
        db.execute("INSERT INTO agent_calls VALUES(?,?,?,?,?,?,?,?,?)",
                   (operation_id, task_key, run_id, category, "reserved", limit, None, time.time_ns(), canonical(payload)))
    body = {"model": config.planner.model, "temperature": 0.2,
            "messages": messages, "max_tokens": min(limit - prompt_reserve, 1200),
            "reasoning_effort": "none"}
    request = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=canonical(body).encode(),
                                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key}, method="POST")
    actual = None
    try:
        envelope = json.loads(transport(request, config.planner.timeout_seconds))
        actual = envelope["usage"]["total_tokens"]
        if type(actual) is not int or not 0 < actual <= limit:
            actual = None
            raise GateError("AGENT_TOKEN_BUDGET")
        if envelope["choices"][0].get("finish_reason") == "length":
            raise GateError("AGENT_RESPONSE_INCOMPLETE")
        response = json.loads(envelope["choices"][0]["message"]["content"])
        if not isinstance(response, dict):
            raise GateError("AGENT_RESPONSE_SCHEMA")
    except Exception:
        with store.transaction() as db:
            db.execute("UPDATE agent_calls SET status='failed',actual_tokens=? WHERE operation_id=?", (actual, operation_id))
        raise GateError("AGENT_CALL_FAILED") from None
    with store.transaction() as db:
        payload["response"] = response
        db.execute("UPDATE agent_calls SET status='completed',actual_tokens=?,payload=? WHERE operation_id=?",
                   (actual, canonical(payload), operation_id))
    return response


def explain_card(store, config, task_key, review_id, *, transport=request_chat_bytes):
    from .reviews import get_card, validate_explanation, summary_facts, append_explanation
    card = get_card(store, task_key, review_id)
    if card["llm_status"] not in {"pending", "queued"}:
        return card
    # Bounded field references; never send full folds, raw samples, paths, or log text.
    summary = {"candidate_id": card["candidate_id"], "stage": card["stage"],
               "evidence_refs": card["evidence_refs"], "facts": summary_facts(card),
               "program_state": card["program_state"], "adaptation_mode": card["adaptation_mode"]}
    try:
        if card["stage"] == "final_test":
            raise GateError("FINAL_TEST_LLM_EXCLUDED")
        response = call_agent(store, config, task_key, card["run_id"], "review", summary,
                              operation_id="explain:" + review_id, transport=transport)
        validate_explanation(response, {**card, "facts": summary["facts"]})
        state, explanation = "generated", response
    except GateError as error:
        if str(error) == "LLM_BUSY":
            return card
        state, explanation = "unavailable:" + str(error), None
    append_explanation(store, task_key, card, state, explanation, config.planner.model)
    return get_card(store, task_key, review_id)
