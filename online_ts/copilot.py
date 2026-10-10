"""Local authenticated conversation and transactional human decisions.

The OS principal is the authentication boundary. Text/JSON can never choose it.
Deploy this CLI only where file access and approved principals are controlled.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
import time
import uuid

from .config import GateError, canonical, digest
from .experiments import capabilities, compile_experiment
from .models import Candidate
from .reviews import get_card, list_cards, render_card, validate_explanation, comparison
from .reviews import summary_facts


def current_actor():
    if os.name == "nt":
        import ctypes
        size = ctypes.c_ulong(257)
        buffer = ctypes.create_unicode_buffer(257)
        if not ctypes.windll.advapi32.GetUserNameW(buffer, ctypes.byref(size)):
            raise GateError("OS_AUTHENTICATION_FAILED")
        return buffer.value
    import pwd
    return pwd.getpwuid(os.getuid()).pw_name


class Copilot:
    def __init__(self, store, artifacts, task_key, session_id=None):
        self.store, self.artifacts, self.key = store, artifacts, store.resolve(task_key)
        self.config, _ = store.task(self.key)
        self.actor = current_actor()
        self.session_id = session_id or "session-" + uuid.uuid4().hex
        with store.transaction() as db:
            row = db.execute("SELECT * FROM human_sessions WHERE session_id=?", (self.session_id,)).fetchone()
            if row and (row["task_key"] != self.key or row["actor"] != self.actor):
                raise GateError("SESSION_NOT_AUTHORIZED")
            db.execute("INSERT OR IGNORE INTO human_sessions VALUES(?,?,?,?)",
                       (self.session_id, self.key, self.actor, canonical({"displayed": [], "final_test_seen": False})))

    def require(self, permission):
        roles = self.config.collaboration.permissions.get(self.actor, ())
        # Read access follows the local database ACL; writes additionally require policy.
        if permission != "read" and permission not in roles:
            raise GateError("HUMAN_PERMISSION_DENIED")

    def _session(self, db):
        return json.loads(db.execute("SELECT payload FROM human_sessions WHERE session_id=?", (self.session_id,)).fetchone()[0])

    def _save_session(self, db, value):
        db.execute("UPDATE human_sessions SET payload=? WHERE session_id=?", (canonical(value), self.session_id))

    def _job(self, run_id):
        job = self.store.job(run_id)
        if job["task_key"] != self.key:
            raise GateError("RUN_NOT_AUTHORIZED")
        return job

    def introduction(self):
        config, state = self.store.task(self.key)
        active = self.store.active_job(self.key)
        return (f"任务 {config.task_id}：预测 {', '.join(config.variables)}，每步 {config.frequency_seconds} 秒，"
                f"预测 {config.horizon_steps} 步。\n当前阶段：{state['status']}；"
                f"训练上限 {config.budget.max_candidates} 个候选、{config.budget.max_training_seconds} 秒。\n"
                "先检验时间回测与低成本基线；合法实验需通过完整配置与资源校验。"
                "业务偏好形成草案，当前 Gate 不会随对话改变。\n"
                + (f"运行 {active['run_id']}：已完成候选 {active['checkpoint']['index']}，"
                   f"已计费训练时间 {active['spent']:.3f} 秒。" if active else "当前无运行中训练。"))

    def models(self, history=False):
        # Migrate existing result evidence to cards idempotently, without training/LLM calls.
        from .reviews import persist_job_cards
        with self.store.transaction() as db:
            for row in db.execute("SELECT * FROM jobs WHERE task_key=?", (self.key,)).fetchall():
                job = self.store._job(row)
                persist_job_cards(db, job, job["checkpoint"], job["result"], job["status"])
        cards = list_cards(self.store, self.key, history=history)
        # Default list excludes sealed Test details. Viewing one is explicit and audited.
        with self.store.transaction() as db:
            session = self._session(db)
            session["displayed"] = [{"review_id": c["review_id"], "candidate_id": c["candidate_id"],
                                      "evidence_version": c["evidence_version"]} for c in cards]
            self._save_session(db, session)
        return [({k: c[k] for k in ("review_id", "run_id", "candidate_id", "candidate", "stage", "evidence_version")}
                 if c["stage"] == "final_test" else c) for c in cards]

    def view(self, review_id, *, explain=False, transport=None):
        card = get_card(self.store, self.key, review_id)
        if card["stage"] == "final_test":
            self.require("release")
            job = self._job(card["run_id"])
            result = job["result"] or {}
            with self.store.transaction() as db:
                session = self._session(db)
                session["final_test_seen"] = True
                self._save_session(db, session)
                db.execute("INSERT OR IGNORE INTO test_exposures VALUES(?,?,?,?,?,?)",
                           (digest(asdict(self.config.source)), result["test_start"], result["test_end"],
                            self.actor, review_id, time.time_ns()))
                self.store._event(db, self.key, "FINAL_TEST_VIEWED", {"actor": self.actor, "review_id": review_id}, time.time_ns())
        if explain:
            from .reviews import queue_explanation
            card = queue_explanation(self.store, self.key, review_id)
            if card["llm_status"] == "queued" and not self.store.active_job(self.key):
                import subprocess
                import sys
                from pathlib import Path
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
                subprocess.Popen([sys.executable, "-B", "-m", "online_ts.review_worker", str(self.store.path), self.key],
                                 env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return card

    def compare(self, left, right):
        return comparison(self.view(left), self.view(right))

    def ask(self, question, review_id, *, transport=None):
        from .agent import call_agent
        card = self.view(review_id)
        if card["stage"] == "final_test":
            return {"status": "explanation_unavailable", "reason": "FINAL_TEST_LLM_EXCLUDED",
                    "deterministic_card": render_card(card)}
        if not isinstance(question, str) or not question.strip() or len(question) > 1200:
            raise GateError("QUESTION_LENGTH")
        evidence = {"mode": "question", "question": question, "facts": summary_facts(card, 12),
                    "evidence_refs": card["evidence_refs"], "program_state": card["program_state"]}
        kwargs = {} if transport is None else {"transport": transport}
        operation = "question:" + digest({"session": self.session_id, "review": review_id, "question": question})
        try:
            response = call_agent(self.store, self.config, self.key, card["run_id"], "question", evidence,
                                  operation_id=operation, **kwargs)
            validate_explanation(response, {**card, "facts": evidence["facts"]})
            answer = {"status": "answered", "explanation": response, "review_id": review_id,
                      "evidence_version": card["evidence_version"], "program_state": card["program_state"],
                      "current_for_stage": get_card(self.store, self.key, review_id)["current_for_stage"]}
        except GateError as error:
            answer = {"status": "explanation_unavailable", "reason": str(error),
                      "deterministic_card": render_card(card)}
        self.store.event(self.key, "HUMAN_QUESTION", {"session_id": self.session_id, "actor": self.actor,
                        "question": question, "answer": answer}, time.time_ns(), operation)
        return answer

    def _experiment_scope(self, job):
        cp = job["checkpoint"]
        if job["status"] != "PENDING" or cp["selected"] is not None or cp.get("feedback_pending") is not None or job["plan"].get("needs_proposal"):
            raise GateError("EXPERIMENT_REQUIRES_UNSEALED_IDLE_RUN")
        meta = self.artifacts.read_json(job["snapshot"])
        start, end = meta["times"][-self.config.search.test_steps], meta["times"][-1]
        with self.store.connection() as db:
            overlap = db.execute("SELECT 1 FROM test_exposures WHERE source_id=? AND test_start<=? AND test_end>=?",
                                 (digest(asdict(self.config.source)), end, start)).fetchone()
            seen = self._session(db)["final_test_seen"]
        if overlap or seen:
            raise GateError("FINAL_TEST_INDEPENDENT_EVALUATION_REQUIRED")

    def draft(self, run_id, *, kind, summary, experiments=(), knowledge=None):
        job = self._job(run_id)
        if kind not in {"experiment", "knowledge", "policy_change", "clarification", "stop"}:
            raise GateError("HUMAN_INTENT_KIND")
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 1200:
            raise GateError("HUMAN_INTENT_SUMMARY")
        if kind == "experiment":
            self.require("experiment")
            self._experiment_scope(job)
            if job["checkpoint"].get("human_pending"):
                raise GateError("HUMAN_DECISION_PENDING")
        elif experiments:
            raise GateError("HUMAN_INTENT_SCHEMA")
        evidence_refs = {job["snapshot"]["sha256"][:24]}
        evidence_refs.update(o["message_id"] for o in job["checkpoint"].get("outcomes", [])
                             if o["evidence_scope"] != "final_test")
        compiled = []
        if kind == "experiment":
            if not isinstance(experiments, (list, tuple)) or not experiments:
                raise GateError("HUMAN_EXPERIMENTS")
            for spec in experiments:
                candidate, target, saved = compile_experiment(
                    spec, self.config, evidence_refs=evidence_refs,
                    available_targets=set(self.config.execution.allowed_targets),
                    parent_ids=[Candidate.from_dict(c).candidate_id for c in job["plan"]["candidates"]])
                compiled.append(saved)
            if len({s["resources"]["training_target"] for s in compiled}) != 1:
                raise GateError("EXPERIMENT_TARGET_CONFLICT")
            prefix = job["plan"]["candidates"][:job["checkpoint"]["index"]]
            ids = [Candidate.from_dict(c).candidate_id for c in prefix] + [Candidate.from_dict(s["candidate"]).candidate_id for s in compiled]
            if len(set(ids)) != len(ids):
                raise GateError("NO_PROGRESS")
            if len(ids) > self.config.budget.max_candidates:
                raise GateError("CANDIDATE_BUDGET")
        if knowledge is not None:
            if (not isinstance(knowledge, dict) or set(knowledge) != {"text", "category", "scope", "source"}
                    or knowledge["category"] not in {"preference", "hypothesis", "business_fact", "annotation", "opinion"}
                    or any(not isinstance(v, str) or not v.strip() or len(v) > 1200 for v in knowledge.values())):
                raise GateError("HUMAN_KNOWLEDGE_SCHEMA")
            knowledge = {**knowledge, "confirmation_status": "unconfirmed", "provider": self.actor}
        binding = digest({"plan": job["plan"], "index": job["checkpoint"]["index"],
                          "reports": job["checkpoint"]["reports"], "policy": self.config.version})
        value = {"schema_version": 1, "kind": kind, "summary": summary, "run_id": run_id,
                 "plan_version": job["checkpoint"]["plan_version"], "policy_version": self.config.version,
                 "actor": self.actor, "session_id": self.session_id, "binding": binding,
                 "experiments": compiled, "knowledge": knowledge, "evidence_refs": sorted(evidence_refs),
                 "original_candidates": job["plan"]["candidates"], "created_at": time.time_ns(),
                 "required_permission": "experiment" if kind == "experiment" else "knowledge"}
        identity = {k: v for k, v in value.items() if k != "created_at"}
        intent_id = "intent-" + digest(identity)[:24]
        value["intent_id"] = intent_id
        with self.store.transaction() as db:
            if kind == "experiment":
                current = self.store._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
                if current["status"] != "PENDING" or current["checkpoint"] != job["checkpoint"] or current["plan"] != job["plan"]:
                    raise GateError("STALE_HUMAN_INTENT")
                cp = current["checkpoint"]
                cp["human_pending"] = intent_id
                db.execute("UPDATE jobs SET checkpoint=? WHERE run_id=?", (canonical(cp), run_id))
            db.execute("INSERT OR IGNORE INTO human_intents VALUES(?,?,?,?,?,?,?)",
                       (intent_id, self.key, self.session_id, run_id, "pending", value["created_at"], canonical(value)))
            self.store._event(db, self.key, "HUMAN_INTENT_DRAFTED", value, value["created_at"], "draft:" + intent_id)
        return value

    def interpret(self, run_id, text, *, transport=None):
        from .agent import call_agent
        job = self._job(run_id)
        self._experiment_scope(job)  # Never send Test-informed conversation to the Planner.
        if not isinstance(text, str) or not text.strip() or len(text) > 1200:
            raise GateError("HUMAN_TEXT_LENGTH")
        import re
        if re.search(r"第二个|刚才那个|用.*那个|\bsecond\b", text, re.IGNORECASE):
            with self.store.connection() as db:
                displayed = self._session(db)["displayed"]
            target = displayed[1] if "第二个" in text and len(displayed) > 1 else None
            summary = ("请确认动作类型及稳定候选 ID/版本；此表达不是发布授权。"
                       + ("当前第二项为 " + target["candidate_id"] + "，证据版本 " + target["evidence_version"] if target else "对象无法唯一确定。"))
            return self.draft(run_id, kind="clarification", summary=summary)
        if re.search(r"final.?test|holdout|最终测试|最终准入|测试集", text, re.IGNORECASE):
            self.store.event(self.key, "TEST_FEEDBACK_REFUSED", {"session_id": self.session_id,
                             "run_id": run_id, "actor": self.actor}, time.time_ns())
            raise GateError("FINAL_TEST_INDEPENDENT_EVALUATION_REQUIRED")
        evidence = {"mode": "intent", "text": text, "capabilities": capabilities(self.config),
                    "evidence_refs": [job["snapshot"]["sha256"][:24]],
                    "current_candidates": job["plan"]["candidates"][-2:],
                    "experiment_schema": {"objective": "goal", "hypothesis": "testable cause",
                        "candidate": {"family": "ridge", "input_steps": self.config.search.inputs[0],
                                      "history_steps": self.config.search.histories[0]},
                        "contract_ref": self.config.version, "evaluation_protocol": "rolling-origin-v1",
                        "resources": {"training_target": "local", "seconds": self.config.budget.max_candidate_seconds,
                                      "memory_mb": self.config.budget.max_memory_mb},
                        "evidence_refs": [job["snapshot"]["sha256"][:24]]}}
        # Approved records remain explicitly typed preferences/hypotheses, never labels.
        evidence["human_context"] = [{"intent_id": v["intent_id"], "category": v["knowledge"]["category"],
                                      "text": v["knowledge"]["text"][:400], "truth_status": "human_statement"}
                                     for v in self.history() if v["status"] == "approved" and v.get("knowledge")][-4:]
        kwargs = {} if transport is None else {"transport": transport}
        response = call_agent(self.store, self.config, self.key, run_id, "intent", evidence,
                              operation_id="intent-call:" + digest({"session": self.session_id, "text": text,
                                  "run": run_id, "version": job["checkpoint"]["plan_version"]}), **kwargs)
        current = self._job(run_id)
        if current["plan"] != job["plan"] or current["checkpoint"] != job["checkpoint"]:
            raise GateError("STALE_HUMAN_INTENT")
        if set(response) != {"kind", "summary", "experiments", "knowledge"}:
            raise GateError("HUMAN_INTENT_SCHEMA")
        return self.draft(run_id, **response)

    def decide(self, intent_id, decision):
        if decision not in {"approve", "reject"}:
            raise GateError("HUMAN_DECISION")
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM human_intents WHERE intent_id=? AND task_key=? AND session_id=?",
                             (intent_id, self.key, self.session_id)).fetchone()
            if row is None:
                raise GateError("INTENT_NOT_AUTHORIZED")
            intent = json.loads(row["payload"])
            self.require(intent["required_permission"])
            if row["status"] != "pending":
                if row["status"] == {"approve": "approved", "reject": "rejected"}[decision]:
                    return {"intent_id": intent_id, "status": row["status"], "duplicate": True}
                raise GateError("HUMAN_DECISION_CONFLICT")
            if decision == "approve":
                job = self.store._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (row["run_id"],)).fetchone())
                binding = digest({"plan": job["plan"], "index": job["checkpoint"]["index"],
                                  "reports": job["checkpoint"]["reports"], "policy": self.config.version})
                if intent["kind"] != "release" and (binding != intent["binding"] or job["checkpoint"]["plan_version"] != intent["plan_version"]):
                    raise GateError("STALE_HUMAN_INTENT")
                if intent["kind"] == "experiment":
                    self._experiment_scope(job)
                    cp = job["checkpoint"]
                    if cp.get("human_pending") != intent_id:
                        raise GateError("STALE_HUMAN_INTENT")
                    from .context import training_seconds_spent
                    if training_seconds_spent(job) >= self.config.budget.max_training_seconds:
                        raise GateError("TRAINING_BUDGET_EXHAUSTED")
                    prefix = job["plan"]["candidates"][:cp["index"]]
                    new = []
                    for spec in intent["experiments"]:
                        raw = {k: v for k, v in spec.items() if k not in {"effective_candidate", "spec_id"}}
                        candidate, target, _ = compile_experiment(raw, self.config,
                            evidence_refs=intent["evidence_refs"], available_targets=set(self.config.execution.allowed_targets),
                            parent_ids=[Candidate.from_dict(c).candidate_id for c in job["plan"]["candidates"]])
                        new.append(asdict(candidate))
                    plan = {**job["plan"], "candidates": prefix + new, "training_target": target,
                            "parent_plan_version": cp["plan_version"], "plan_version": cp["plan_version"] + 1,
                            "experiment_history": job["plan"].get("experiment_history", []) + intent["experiments"]}
                    from .planner import proposal_fingerprint
                    plan["proposal_fingerprint"] = proposal_fingerprint(plan)
                    cp.update(plan_version=plan["plan_version"], selected=None, human_pending=None)
                    cp.pop("selection_binding", None)
                    db.execute("UPDATE jobs SET plan=?,checkpoint=? WHERE run_id=?",
                               (canonical(plan), canonical(cp), row["run_id"]))
                    db.execute("INSERT OR IGNORE INTO proposal_history VALUES(?,?,?,?,?)",
                               (self.key, plan["proposal_fingerprint"], row["run_id"], time.time_ns(), "accepted"))
                elif intent["kind"] == "release":
                    row_state = db.execute("SELECT state FROM tasks WHERE task_key=?", (self.key,)).fetchone()
                    state = json.loads(row_state[0])
                    if state["stop_requested"] or state["stage"] not in {"INITIAL_APPROVAL", "CANARY"}:
                        raise GateError("NO_PENDING_APPROVAL")
                    release = self.store._release_binding(db, self.key, state, self.config)
                    if canonical(release) != canonical(intent["binding"]):
                        raise GateError("STALE_RELEASE_CONFIRMATION")
                    state.update(approved=True, approval_binding=release)
                    db.execute("UPDATE tasks SET state=?,revision=revision+1 WHERE task_key=?", (canonical(state), self.key))
                    self.store._event(db, self.key, "APPROVED", {"actor": self.actor, "binding": release}, time.time_ns(), "release:" + intent_id)
                elif intent["kind"] == "knowledge":
                    if intent["knowledge"] is None:
                        raise GateError("HUMAN_KNOWLEDGE_REQUIRED")
                    # Approval confirms recording only; it never creates labels or features.
                elif intent["kind"] in {"policy_change", "clarification", "stop"}:
                    raise GateError("EXPLICIT_SEPARATE_OPERATION_REQUIRED")
            state = "approved" if decision == "approve" else "rejected"
            if decision == "reject" and intent["kind"] == "experiment":
                current = self.store._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (row["run_id"],)).fetchone())
                cp = current["checkpoint"]
                if cp.get("human_pending") == intent_id:
                    cp["human_pending"] = None
                    db.execute("UPDATE jobs SET checkpoint=? WHERE run_id=?", (canonical(cp), row["run_id"]))
            db.execute("UPDATE human_intents SET status=? WHERE intent_id=?", (state, intent_id))
            result = {"schema_version": 1, "intent_id": intent_id, "decision": decision, "status": state,
                      "actor": self.actor, "plan_version": intent["plan_version"],
                      "policy_version": self.config.version, "created_at": time.time_ns()}
            self.store._event(db, self.key, "HUMAN_DECISION", result, result["created_at"], "decision:" + intent_id)
            return result

    def release_draft(self):
        self.require("release")
        binding = self.store.release_binding(self.key)
        model = self.store.model(binding["model_id"])
        value = {"schema_version": 1, "kind": "release", "summary": "确认具体发布候选；程序仍须重验 Gate。",
                 "run_id": model["run_id"], "plan_version": binding["plan_version"],
                 "policy_version": self.config.version, "actor": self.actor,
                 "session_id": self.session_id, "binding": binding, "experiments": [],
                 "knowledge": None, "evidence_refs": ["artifact:" + binding["model_package_hash"]],
                 "required_permission": "release", "created_at": time.time_ns()}
        intent_id = "release-intent-" + digest({"binding": binding, "session": self.session_id})[:24]
        value["intent_id"] = intent_id
        with self.store.transaction() as db:
            db.execute("INSERT OR IGNORE INTO human_intents VALUES(?,?,?,?,?,?,?)",
                       (intent_id, self.key, self.session_id, model["run_id"], "pending", value["created_at"], canonical(value)))
            self.store._event(db, self.key, "RELEASE_CONFIRMATION_REQUESTED", value, value["created_at"], "draft:" + intent_id)
        return value

    def history(self):
        with self.store.connection() as db:
            return [{"status": row["status"], **json.loads(row["payload"])} for row in db.execute(
                "SELECT status,payload FROM human_intents WHERE task_key=? AND session_id=? ORDER BY created_ns", (self.key, self.session_id))]


def render_draft(value):
    lines = [f"草案 {value['intent_id']} | {value['kind']}", value["summary"],
             f"计划版本 {value['plan_version']}；权限 {value['required_permission']}；尚未执行。"]
    for spec in value["experiments"]:
        lines.extend(["假设：" + spec["hypothesis"], "实际候选：" + canonical(spec["effective_candidate"]),
                      "请求资源：" + canonical(spec["resources"])])
    if value["knowledge"]:
        lines.append("人工知识（未确认为训练真值）：" + canonical(value["knowledge"]))
    if value["kind"] == "release":
        lines.append("发布确认绑定：" + canonical(value["binding"]))
    else:
        lines.append("审批仅适用于此草案；不等于批准上线。")
    return "\n".join(lines)
