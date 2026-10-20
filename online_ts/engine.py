from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .config import GateError, SECOND, TaskConfig, digest
from .data import ArtifactStore, InfluxSource, ReplaySource, aligned, causal_fill, snapshot
from .models import load_forecast_model
from .monitoring import degradation, enough, paired_comparison, summarize
from .planner import make_proposal, proposal_fingerprint, validate_proposal
from .store import Store
from .training import choose
from .labels import issue_window


class Engine:
    def __init__(self, runtime: str | Path, key: str, *, source=None):
        self.runtime = Path(runtime).resolve()
        self.store = Store(self.runtime / "lifecycle.db")
        self.artifacts = ArtifactStore(self.runtime / "artifacts")
        self.key = self.store.resolve(key)
        self.config, state = self.store.task(self.key)
        self.owner = uuid.uuid4().hex
        self._source = source
        self.processes: dict[str, tuple[subprocess.Popen, str]] = {}
        self._models: OrderedDict[str, object] = OrderedDict()
        self._monitor_inputs: dict[tuple[str, int], tuple] = {}
        self.review_process = None
        self.review_disabled = False

    @property
    def source(self):
        # Management operations depend only on durable state, even if the source
        # artifact is corrupt or the source service is unavailable.
        if self._source is None:
            if self.config.source.kind == "history_http":
                from .history_source import HistorySource
                self._source = HistorySource(self.config)
                return self._source
            self._source = (ReplaySource(self.config, self.artifacts, self._state()["source_ref"])
                            if self.config.source.kind == "file" else InfluxSource(self.config))
        return self._source

    @classmethod
    def create(cls, runtime: str | Path, config: TaskConfig, *, source=None) -> "Engine":
        root = Path(runtime).resolve()
        store = Store(root / "lifecycle.db")
        key = Store.task_key(config)
        try:
            _, existing = store.task(key)
        except GateError:
            existing = None
        adapter = source
        ref = existing["source_ref"] if existing else None
        if config.source.kind == "file" and source is None:
            adapter = ReplaySource(config, ArtifactStore(root / "artifacts"), ref)
            ref = adapter.ref
        key = store.create_task(config, ref)
        return cls(root, key, source=adapter)

    def _state(self) -> dict:
        return self.store.task(self.key)[1]

    def _load_model(self, model_id: str):
        entry = self.store.model(model_id)
        if entry["task_key"] != self.key:
            raise GateError("MODEL_TASK_SCOPE")
        # The small content-addressed wrapper is reverified on every use; heavyweight
        # weights/pipelines stay cached by immutable registry version.
        self.artifacts.read_json(entry["artifact"])
        if model_id in self._models:
            self._models.move_to_end(model_id)
            return self._models[model_id]
        try:
            model = load_forecast_model(self.artifacts, entry["artifact"], self.config)
            probe = self.artifacts.read_json(entry["report"]["reload_probe"])
            tolerance = 1e-5 if model.candidate.family == "chronos2" else 1e-10
            if not np.allclose(model.predict(np.asarray(probe["input"])), probe["prediction"], rtol=tolerance, atol=tolerance):
                raise GateError("MODEL_RELOAD_MISMATCH")
        except GateError:
            raise
        except (ValueError, KeyError, IndexError, TypeError, OverflowError):
            raise GateError("MODEL_PACKAGE_INVALID") from None
        self._models[model_id] = model
        maximum = self.config.foundation.adapter_cache_size if self.config.foundation.enabled else 2
        while len(self._models) > maximum:
            self._models.popitem(last=False)
        return model

    def _launch(self, job: dict):
        owner = uuid.uuid4().hex
        reserved = self.store.reserve_stage(job["run_id"], owner, self.config)
        if reserved is None:
            return
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        command = [sys.executable, "-B", "-m", "online_ts.worker", str(self.store.path), str(self.artifacts.root), job["run_id"], owner]
        try:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError:
            self.store.fail_stage(job["run_id"], owner, "WORKER_START_FAILED", self.config, retryable=True)
            return
        self.processes[job["run_id"]] = (process, owner)

    def pump_training(self):
        """Nonblocking. Executable work occurs only in a bounded child process."""
        if not self.review_disabled:
            try:
                self._pump_reviews()
            except Exception:
                # Optional explanations must not block stop, serving, or training.
                self.review_disabled = True
        state = self._state()
        for run_id, (process, owner) in list(self.processes.items()):
            job = self.store.job(run_id)
            timed_out = job["status"] == "RUNNING" and time.time() - job["started_wall"] >= job["reservation"]
            if state["stop_requested"] or timed_out:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                if state["stop_requested"]:
                    self.store.cancel_job(run_id)
                else:
                    self.store.fail_stage(run_id, owner, "TRAINING_DEADLINE", self.config, retryable=True)
                self.processes.pop(run_id, None)
            elif process.poll() is not None:
                process.wait()
                if job["status"] == "RUNNING":
                    self.store.fail_stage(run_id, owner, "WORKER_INTERRUPTED", self.config, retryable=True)
                self.processes.pop(run_id, None)
        job = self.store.active_job(self.key)
        if state["stop_requested"]:
            if job:
                self.store.cancel_job(job["run_id"])
            return
        if not job:
            return
        if job["run_id"] in self.processes:
            # A worker may commit its checkpoint just before its interpreter exits.
            # Reap that process before replacing the handle or launching the next stage.
            return
        if job["status"] == "RUNNING":
            if job["lease_until"] < time.time():
                # Its full reservation remains charged. A stale worker cannot commit after fencing.
                self.store.fail_stage(job["run_id"], job["owner"], "WORKER_LEASE_EXPIRED", self.config, retryable=True)
            return
        cp = job["checkpoint"]
        if cp.get("human_pending"):
            return
        if (not job["plan"].get("needs_proposal") and cp.get("feedback_pending") is None
                and cp["selected"] is None and cp["index"] >= len(job["plan"]["candidates"])):
            if job["plan"].get("agent_managed"):
                self.store.event(self.key, "AGENT_EXPERIMENT_READY", {"run_id": job["run_id"]},
                                 state["now"] or job["cutoff"], "agent-ready:" + job["run_id"])
                return
            try:
                self.store.seal_selection(job["run_id"], choose(cp["reports"], self.config, artifacts=self.artifacts, snapshot_sha=job["snapshot"]["sha256"]))
            except GateError as error:
                self.store.seal_selection(job["run_id"], None, str(error))
            job = self.store.job(job["run_id"])
        if job["status"] == "PENDING":
            self._launch(job)

    def _consume_training(self, now: int, *, agent_run_id: str | None = None):
        # Consume every durable terminal job in cutoff order. A per-job disposition is the
        # acknowledgement; the legacy pointer remains informational only.
        for job in self.store.terminal_jobs_without_disposition(self.key):
            if job["plan"].get("agent_managed") and job["run_id"] != agent_run_id:
                continue
            run_id, result = job["run_id"], job["result"]
            if result and result.get("waiting_for_labels"):
                from .training import evaluate_frozen_final
                try:
                    truth = self.source.labels_until(now)
                    result = evaluate_frozen_final(result, truth, now, self.config, self.artifacts)
                except (GateError, AttributeError):
                    continue
                if result.get("waiting_for_labels"):
                    self.store.change(self.key, lambda s: s.update(status="WAITING_FOR_LABELS"))
                    continue
                self.store.complete_waiting_evaluation(run_id, result, now)

            def settled(state: dict) -> None:
                state["last_consumed_job"] = run_id
                if not state["stop_requested"]:
                    state["cooldown_until"] = now + int(self.config.monitoring.cooldown_seconds * SECOND)
                    state["status"] = "SERVING" if state["champion"] else "DATA_BLOCKED"

            if job["status"] != "COMPLETED" or not result or not result.get("passed"):
                issues = result.get("issues", [result.get("code")]) if result else [job["status"]]
                outcome = "cancelled" if job["status"] == "CANCELLED" else "failed" if job["status"] == "FAILED" else "rejected"
                self.store.dispose_job(run_id, outcome, {"issues": issues}, now, settled)
                continue
            try:
                model_id = self.store.register_model(self.key, run_id, result, now)
                self._load_model(model_id)
            except (GateError, KeyError, TypeError, ValueError):
                self.store.dispose_job(run_id, "rejected", {"issues": ["RELEASE_PACKAGE_INVALID"]}, now, settled)
                continue
            state = self._state()
            if state["champion"] == model_id:
                self.store.dispose_job(run_id, "registered", {"model_id": model_id, "action": "already_champion"}, now,
                                       lambda s: s.update(last_consumed_job=run_id))
            elif state["challenger"] and state["challenger"] != model_id:
                self.store.dispose_job(run_id, "rejected", {"issues": ["ACTIVE_CHALLENGER"], "model_id": model_id}, now, settled)
            elif state["champion"] is None:
                if self.config.mode == "live":
                    def initial_approval(s: dict) -> None:
                        s.update(challenger=model_id, stage="INITIAL_APPROVAL", stage_since=now,
                                 last_consumed_job=run_id, status="AWAITING_APPROVAL")
                    self.store.dispose_job(run_id, "registered", {"model_id": model_id, "action": "awaiting_initial_approval"}, now, initial_approval)
                    self.store.event(self.key, "INITIAL_APPROVAL_REQUIRED", {"model_id": model_id}, now, "approval:" + run_id)
                else:
                    self.store.switch(self.key, None, model_id, now,
                                      {"passed": True, "kind": "INITIAL_OFFLINE_GATE", "run_id": run_id}, "initial:" + model_id)
                    self.store.dispose_job(run_id, "registered", {"model_id": model_id, "action": "initial_champion"}, now,
                                           lambda s: s.update(last_consumed_job=run_id))
            else:
                def shadow(s: dict) -> None:
                    s.update(challenger=model_id, stage="SHADOW", stage_since=now, approved=False,
                             last_consumed_job=run_id, status="SHADOWING")
                self.store.dispose_job(run_id, "registered", {"model_id": model_id, "action": "shadow"}, now, shadow)
                self.store.event(self.key, "SHADOW_STARTED", {"model_id": model_id, "champion": state["champion"]}, now, "shadow:" + run_id)

    def _pump_reviews(self):
        if self.review_process is not None:
            if self.review_process.poll() is None:
                return
            self.review_process.wait()
            self.review_process = None
        if (self.config.planner.kind != "external" or self._state()["stop_requested"]
                or self._state().get("decision_owner", "workflow") != "workflow"):
            return
        from .reviews import list_cards
        if not any(c["stage"] != "final_test" and (c["llm_status"] == "queued" or
                   (self.config.collaboration.automatic_reviews and c["llm_status"] == "pending"))
                   for c in list_cards(self.store, self.key)):
            return
        if self.store.active_job(self.key):
            return  # Planning/feedback get the single LLM slot before optional prose.
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        try:
            self.review_process = subprocess.Popen(
                [sys.executable, "-B", "-m", "online_ts.review_worker", str(self.store.path), self.key],
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError:
            self.store.event(self.key, "MODEL_REVIEW_WORKER_UNAVAILABLE", {}, time.time_ns())

    def _request_training(self, frame, cutoff: int, now: int, reason: str) -> str | None:
        state = self._state()
        if state.get("decision_owner", "workflow") != "workflow":
            kind = "MODEL_DEGRADATION_DETECTED" if reason == "PERSISTENT_DEGRADATION" else "MODEL_AGE_THRESHOLD"
            incident = state["monitor"].get("alert_id") if reason == "PERSISTENT_DEGRADATION" else state.get("model_since")
            self.store.event(self.key, kind, {"reason": reason, "model_id": state.get("champion")}, now,
                             f"agent-trigger:{reason}:{state.get('champion')}:{incident}:{state.get('decision_epoch')}")
            return None
        if any((job["result"] or {}).get("waiting_for_labels")
               for job in self.store.terminal_jobs_without_disposition(self.key)):
            return None
        if state["stop_requested"] or state["challenger"] or now < state["cooldown_until"] or self.store.active_job(self.key):
            return None
        eligible = frame.loc[frame.index <= cutoff]
        maturity = frame.attrs.get("label_availability_by_time", {})
        mature_times = [int(t) for t in eligible.index if maturity.get(int(t), int(t)) <= now]
        if mature_times:
            eligible = eligible.loc[eligible.index <= max(mature_times)]
        else:
            raise GateError("TRAINING_LABELS_INSUFFICIENT")
        if eligible.empty:
            raise GateError("TRAINING_LABELS_INSUFFICIENT")
        cutoff = min(cutoff, int(eligible.index[-1]))
        if state["last_train_cutoff"] is not None and cutoff - state["last_train_cutoff"] < self.config.monitoring.minimum_new_steps * self.config.step_ns:
            return None
        history = aligned(eligible, self.config, end=cutoff).tail(self.config.search.required_history)
        if len(history) < self.config.search.required_history:
            raise GateError("INITIAL_HISTORY_INSUFFICIENT")
        ref = snapshot(history, cutoff, self.config, self.artifacts, as_of=now).to_dict()
        summary = self.artifacts.read_json(ref)["profile"]
        if self.config.planner.kind == "external":
            # Network planning is a reserved worker stage; serving never waits for it.
            proposal, tokens = {"snapshot_sha": ref["sha256"], "candidates": [], "needs_proposal": True}, 0
        else:
            proposal, tokens, routing = make_proposal(self.config, ref["sha256"], summary)
            validate_proposal(proposal, self.config, ref["sha256"], self.store.proposal_fingerprints(self.key))
        # The execution envelope adds trusted references after validating the untrusted proposal.
        envelope = dict(proposal)
        if not envelope.get("needs_proposal"):
            envelope["proposal_fingerprint"] = proposal_fingerprint(proposal)
        envelope["planner_tokens"] = tokens
        if self.config.planner.kind != "external":
            envelope["routing_evidence"] = routing
        envelope["champion_artifact"] = self.store.model(state["champion"])["artifact"] if state["champion"] else None
        return self.store.request_job(self.key, cutoff, ref, envelope, reason, now)

    def _reject(self, now: int, code: str, evidence: dict | None = None):
        challenger = self._state()["challenger"]
        self.store.change(self.key, lambda s: s.update(challenger=None, stage=None, stage_since=None, approved=False,
                                                       status="SERVING" if s["champion"] else "DATA_BLOCKED", cooldown_until=now + int(self.config.monitoring.cooldown_seconds * SECOND)),
                          "CHALLENGER_REJECTED", {"model_id": challenger, "code": code, "evidence": evidence}, now)
        if challenger:
            self._models.pop(challenger, None)

    def rollback(self, reason: str = "MANUAL_ROLLBACK", *, operation: str | None = None):
        state = self._state()
        if not state["previous"]:
            raise GateError("NO_ROLLBACK_VERSION")
        self._load_model(state["previous"])
        now = state["now"] or 0
        self.store.switch(self.key, state["champion"], state["previous"], now, {"reason": reason},
                          operation or "rollback:" + digest({"from": state["champion"], "to": state["previous"], "since": state["model_since"]}), rollback=True)

    def approve(self, actor: str):
        if not actor.strip() or len(actor) > 120:
            raise GateError("APPROVER_REQUIRED")
        state = self._state()
        if not state["challenger"] or state["stage"] not in {"INITIAL_APPROVAL", "CANARY"}:
            raise GateError("NO_PENDING_APPROVAL")
        self._load_model(state["challenger"])
        from .copilot import current_actor
        self.store.approve_release(self.key, current_actor())
        if state["stage"] == "INITIAL_APPROVAL":
            self.store.switch(self.key, None, state["challenger"], state["now"] or state["stage_since"], {"passed": True, "kind": "INITIAL_OFFLINE_GATE"}, "initial:" + state["challenger"])

    def _release_gate(self, now: int, watermark: int, *, agent_operation: str | None = None):
        state = self._state()
        if not state["challenger"] or state["stage"] == "INITIAL_APPROVAL":
            return
        start = state["stage_since"]
        if now - start > self.config.release.maximum_observation_seconds * SECOND:
            self._reject(now, "OBSERVATION_TIMEOUT")
            return
        duration = self.config.release.shadow_seconds if state["stage"] == "SHADOW" else self.config.release.canary_seconds
        if now - start < duration * SECOND:
            return
        champion_model = self._load_model(state["champion"])
        bounds = self.store.prediction_issue_bounds(self.key, state["champion"])
        expected = issue_window(start, int(duration * SECOND), self.config.step_ns, bounds[0] if bounds else start)
        left = self.store.prediction_rows_by_issue(self.key, state["champion"], expected.start, expected.stop, now)
        right = self.store.prediction_rows_by_issue(self.key, state["challenger"], expected.start, expected.stop, now)
        evidence = paired_comparison(left, right, self.config, champion_model.scale, expected_origins=expected)
        identity = digest({"stage_start": start, "labels": [(r["issue_ns"], r["target_ns"], r["variable"], r.get("label_revision")) for r in right]})
        self.store.event(self.key, "PAIRED_GATE", {"stage": state["stage"], "evidence": evidence}, now, f"pair:{state['challenger']}:{state['stage']}:{identity}")
        if evidence["issues"] == ["PAIRED_LABEL_COVERAGE"]:
            return
        if not evidence["passed"]:
            self._reject(now, "ONLINE_GATE_REJECTED", evidence)
            return
        if state.get("decision_owner", "workflow") == "agent" and agent_operation is None:
            return  # Supervisor may reject unsafe evidence; only Agent may advance.
        if state["stage"] == "SHADOW":
            self.store.change(self.key, lambda s: s.update(stage="CANARY", stage_since=now, status="CANARY"), "CANARY_STARTED", {"model_id": state["challenger"], "evidence": evidence, **({"agent_operation": agent_operation} if agent_operation else {})}, now)
        elif self.config.mode == "live" and not state["approved"]:
            self.store.change(self.key, lambda s: s.update(status="AWAITING_APPROVAL"))
        else:
            if self.config.mode == "live" and state.get("approval_binding") != self.store.release_binding(self.key):
                self.store.change(self.key, lambda s: s.update(approved=False, status="AWAITING_APPROVAL"),
                                  "APPROVAL_EXPIRED", {"model_id": state["challenger"]}, now)
                return
            self._load_model(state["challenger"])
            self.store.switch(self.key, state["champion"], state["challenger"], now, evidence, agent_operation or "promote:" + state["challenger"])

    def _canary_safety_gate(self, now: int, watermark: int) -> bool:
        """Remove Canary traffic on mature critical evidence without waiting for promotion time."""
        state = self._state()
        if state["stage"] != "CANARY" or not state["champion"] or not state["challenger"]:
            return False
        champion_model = self._load_model(state["champion"])
        start = state["stage_since"]
        bounds = self.store.prediction_issue_bounds(self.key, state["champion"])
        expected = issue_window(start, self.config.release.minimum_origins * self.config.step_ns,
                                self.config.step_ns, bounds[0] if bounds else start)
        safety_issues = []
        while bounds and expected.start <= bounds[1]:
            left = self.store.prediction_rows_by_issue(self.key, state["champion"], expected.start, expected.stop, now)
            right = self.store.prediction_rows_by_issue(self.key, state["challenger"], expected.start, expected.stop, now)
            evidence = paired_comparison(left, right, self.config, champion_model.scale, expected_origins=expected)
            safety_issues = sorted(set(evidence["issues"]) & {
                "CRITICAL_VARIABLE_REGRESSION", "MAJORITY_REGRESSION", "INFERENCE_BUDGET",
                "PAIRED_LABEL_MISMATCH", "INPUT_QUALITY", "WORST_BUSINESS_GROUP_REGRESSION",
            })
            if safety_issues:
                evidence.update(safety_window_start=expected.start, safety_window_end=expected.stop)
                break
            expected = range(expected.stop, expected.stop + len(expected) * self.config.step_ns, self.config.step_ns)
        if not safety_issues:
            return False
        triggered = time.perf_counter()
        issue = self._state().get("last_issue")
        if issue is not None:
            self.store.fallback_issue(self.key, issue, state["champion"], state["challenger"], now,
                                      "CANARY_SAFETY_GATE")
        self._reject(now, "CANARY_SAFETY_GATE", {**evidence, "safety_issues": safety_issues})
        elapsed_ms = (time.perf_counter() - triggered) * 1000
        evidence = {**evidence, "safety_issues": safety_issues, "rollback_elapsed_ms": elapsed_ms}
        self.store.event(self.key, "CANARY_SAFETY_ROLLBACK", evidence, now,
                         f"canary-safety:{state['challenger']}:{start}")
        return True

    def _evaluate_mature_windows(self, model_id: str, now: int, *, track_degradation: bool,
                                 history: dict | None = None) -> tuple[dict, bool, list[dict]]:
        """Evaluate every historical prediction window as labels arrive or are revised."""
        model = self._load_model(model_id)
        entry = self.store.model(model_id)
        state_history = dict(history or {})
        degraded = False
        completed_evidence: list[dict] = []
        bounds = self.store.prediction_issue_bounds(self.key, model_id)
        if not bounds:
            return state_history, degraded, completed_evidence
        first_issue, last_issue = bounds
        width = self.config.evaluation_ns
        previous_windows = {(window["window_start"], window["window_end"]): window
                            for window in self.store.monitoring_windows(self.key, model_id)}
        if track_degradation:
            # Window evidence is durable before the task's derived summary.
            # Reconstruct it so a crash between those commits cannot preserve
            # withdrawn evidence or miss an already-committed evaluation.
            state_history = {}
            for _, saved in sorted(previous_windows.items()):
                if saved["label_state"] in {"complete", "revised"}:
                    state_history, degraded = degradation(saved["evidence"], state_history, self.config,
                        offline_reference=entry["report"]["offline"]["normalized_mse"])
        input_counts = self.store.prediction_window_counts(self.key, model_id, first_issue, width, now)
        window_start = first_issue
        ordinal = 0
        while window_start <= last_issue:
            window_end = window_start + width
            previous = previous_windows.get((window_start, window_end))
            window_complete = last_issue >= window_end - self.config.step_ns
            cache_key = (model_id, window_start)
            inputs = (input_counts.get(window_start), window_complete)
            # Both input tables append immutable rows. A fresh Engine has no
            # cache and rechecks all fingerprints, including after a restart.
            if previous and self._monitor_inputs.get(cache_key) == inputs:
                ordinal += 1
                window_start = window_end
                continue
            rows = self.store.prediction_rows_by_issue(self.key, model_id, window_start, window_end, now)
            if rows:
                label_fingerprint = digest([
                    (row["issue_ns"], row["target_ns"], row["variable"],
                     row.get("label_revision"), row["actual"])
                    for row in rows
                ])
                if (previous and previous["label_fingerprint"] == label_fingerprint
                        and previous["evidence"].get("window_complete") == window_complete):
                    self._monitor_inputs[cache_key] = inputs
                    ordinal += 1
                    window_start = window_end
                    continue
                evidence = summarize(rows, self.config, model.scale,
                                     expected_origins=range(window_start, window_end, self.config.step_ns))
                evidence.update(model_id=model_id, window_start=window_start, window_end=window_end,
                                window_complete=window_complete)
                if enough(evidence, self.config) and window_complete:
                    from .monitoring import same_window_baselines
                    evidence["same_window_baselines"] = same_window_baselines(rows, self.source, self.config, model.scale,
                        expected_origins=range(window_start, window_end, self.config.step_ns))
                from .qualification import regime_context
                evidence["regime"] = regime_context(
                    self.source.read_until(now, start=window_start), window_start,
                    window_end + self.config.horizon_steps * self.config.step_ns,
                    declared=bool(self.config.source.regime_column)) if self.config.source.regime_column else {
                        "status": "unobserved", "regime_id": "unlabelled", "source": "not_provided"}
                if evidence["labeled_count"] == 0:
                    label_state = "not_arrived"
                elif enough(evidence, self.config) and evidence["window_complete"]:
                    label_state = ("revised" if previous
                                   and previous["label_state"] in {"complete", "revised"}
                                   and (previous["label_fingerprint"] != label_fingerprint
                                        or previous["label_state"] == "revised")
                                   else "complete")
                else:
                    label_state = "partial"
                evidence["label_state"] = label_state
                changed = self.store.save_monitoring_window(
                    self.key, model_id, window_start, window_end, now,
                    label_state, label_fingerprint, evidence)
                if changed:
                    self.store.save_monitor(self.key, model_id, now + ordinal, evidence)
                    if label_state in {"not_arrived", "partial"}:
                        self.store.event(
                            self.key, "WAITING_FOR_LABELS",
                            {"model_id": model_id, "window_start": window_start,
                             "window_end": window_end, "state": label_state,
                             "coverage": evidence["label_coverage"], "origins": evidence["origins"]},
                            now, f"labels:{model_id}:{window_start}:{label_fingerprint}")
                        if track_degradation and previous and previous["label_state"] in {"complete", "revised"}:
                            # A newly received immature correction retracts the old
                            # truth. Remove this window from degradation evidence too.
                            state_history, degraded = {}, False
                            for key, window in sorted(previous_windows.items()):
                                if key == (window_start, window_end) or window["label_state"] not in {"complete", "revised"}:
                                    continue
                                state_history, degraded = degradation(window["evidence"], state_history, self.config,
                                    offline_reference=entry["report"]["offline"]["normalized_mse"])
                    else:
                        kind = ("MONITORING_WINDOW_REVISED" if label_state == "revised"
                                else "MONITORING_WINDOW_EVALUATED")
                        self.store.event(
                            self.key, kind,
                            {"model_id": model_id, "window_start": window_start,
                             "window_end": window_end, "label_fingerprint": label_fingerprint},
                            now, f"monitor:{model_id}:{window_start}:{label_fingerprint}")
                        if track_degradation:
                            reference_before = state_history.get("active_regime")
                            assessed, assessed_fingerprint = evidence, label_fingerprint
                            if label_state == "revised" or window_start < state_history.get("last_window_end", window_start):
                                # Revisions and late older windows replace evidence, never add independent checks.
                                windows = {**previous_windows, (window_start, window_end): {
                                    "evidence": evidence, "label_state": label_state, "label_fingerprint": label_fingerprint}}
                                state_history = {}
                                for _, window in sorted(windows.items()):
                                    if window["label_state"] not in {"complete", "revised"}:
                                        continue
                                    assessed, assessed_fingerprint = window["evidence"], window["label_fingerprint"]
                                    state_history, this_degraded = degradation(assessed, state_history, self.config,
                                        offline_reference=entry["report"]["offline"]["normalized_mse"])
                            else:
                                state_history, this_degraded = degradation(evidence, state_history, self.config,
                                    offline_reference=entry["report"]["offline"]["normalized_mse"])
                            degraded = this_degraded
                            completed_evidence.append(evidence)
                            if self._state().get("decision_owner") == "agent":
                                from .agent_runtime.observers import record_assessment
                                record_assessment(self, model_id, assessed_fingerprint, assessed, state_history,
                                                  reference_before, this_degraded, now)
                previous_windows[(window_start, window_end)] = {"evidence": evidence, "label_state": label_state,
                    "label_fingerprint": label_fingerprint}
            self._monitor_inputs[cache_key] = inputs
            ordinal += 1
            window_start = window_end
        return state_history, degraded, completed_evidence

    def _monitor(self, now: int, watermark: int, frame):
        state = self._state()
        if state["last_check"] is not None and now - state["last_check"] < self.config.monitoring.check_seconds * SECOND:
            return
        if not state["champion"]:
            return
        entry = self.store.model(state["champion"])
        history, degraded, completed_evidence = self._evaluate_mature_windows(
            state["champion"], now, track_degradation=True, history=state["monitor"])
        # Labels may mature after a model has left service.  Those predictions remain
        # evidence and are evaluated exactly once per label fingerprint.
        for historical_model in self.store.model_ids_with_predictions(self.key):
            if historical_model != state["champion"]:
                try:
                    self._evaluate_mature_windows(historical_model, now, track_degradation=False)
                except GateError as error:
                    self.store.event(self.key, "HISTORICAL_MODEL_UNAVAILABLE", {"model_id": historical_model, "code": str(error)},
                                     now, "historical-unavailable:" + historical_model + ":" + str(error))
        self.store.change(self.key, lambda s: s.update(last_check=now, monitor=history))
        if self._post_promotion_safety_gate(now):
            return
        if self._canary_safety_gate(now, watermark):
            return
        self._release_gate(now, watermark)
        state = self._state()
        if (state.get("observation_until") is not None and now >= state["observation_until"]
                and state.get("observation_completed_for") != state["champion"]):
            bounds = self.store.prediction_issue_bounds(self.key, state["champion"])
            expected_range = issue_window(state["model_since"], state["observation_until"] - state["model_since"],
                                          self.config.step_ns, bounds[0] if bounds else state["model_since"])
            expected = set(expected_range)
            rows = self.store.prediction_rows_by_issue(self.key, state["champion"], expected_range.start, expected_range.stop, now)
            observation = summarize(rows, self.config, self._load_model(state["champion"]).scale, expected_origins=expected)
            origins = sorted({row["issue_ns"] for row in rows})
            observation["duration_elapsed"] = True
            observation["prediction_period_covered"] = bool(expected) and set(origins) == expected
            observation["mature_windows"] = len([window for window in self.store.monitoring_windows(self.key, state["champion"])
                if window["window_start"] >= state["model_since"] and window["label_state"] in {"complete", "revised"}])
            observation["input_fresh"] = state.get("last_issue") is not None and now - state["last_issue"] <= self.config.step_ns
            observation["errors"] = sum(event["kind"] in {"INFERENCE_FAILED", "DATA_OR_WORKFLOW_BLOCKED"}
                and state["model_since"] <= event["event_time"] <= state["observation_until"]
                for event in self.store.events(self.key))
            retained_qualified = True
            if state["previous"]:
                retained_rows = self.store.prediction_rows_by_issue(self.key, state["previous"], expected_range.start, expected_range.stop, now)
                paired = paired_comparison(retained_rows, rows, self.config, self._load_model(state["previous"]).scale, expected_origins=expected)
                retained_qualified = (paired["coverage"] == 1. and enough(paired["champion"], self.config)
                    and enough(paired["challenger"], self.config) and paired["champion"]["clean_inputs"]
                    and not set(paired["issues"]) & {"PAIRED_LABEL_MISMATCH", "CRITICAL_VARIABLE_REGRESSION", "WORST_BUSINESS_GROUP_REGRESSION"})
                observation["retained_model_pair_qualified"] = retained_qualified
            ready = (enough(observation, self.config) and observation["prediction_period_covered"]
                     and observation["mature_windows"] > 0 and observation["input_fresh"]
                     and observation["errors"] == 0 and observation["clean_inputs"] and retained_qualified
                     and observation["maximum_inference_ms"] <= min(self.config.release.maximum_inference_ms,
                         self.config.frequency_seconds * 1000))
            if ready:
                self.store.change(self.key, lambda value: value.update(observation_completed_for=value["champion"]),
                    "POST_PROMOTION_OBSERVATION_COMPLETED", {"model_id": state["champion"], "evidence": observation}, now)
            else:
                exceeded = now - state["model_since"] > self.config.release.maximum_observation_seconds * SECOND
                self.store.event(self.key, "POST_PROMOTION_OBSERVATION_WAITING",
                    {"model_id": state["champion"], "evidence": observation,
                     "maximum_duration_exceeded": exceeded},
                    now, f"observation-wait:{state['champion']}:{state['model_since']}:{observation['mature_windows']}:{int(exceeded)}")
            state = self._state()
        age_due = state["model_since"] is not None and now - state["model_since"] >= self.config.monitoring.max_model_age_seconds * SECOND
        # Age triggers reevaluation even before a full label window; the data gate still applies.
        if degraded or age_due:
            self._request_training(frame, watermark, now, "PERSISTENT_DEGRADATION" if degraded else "MODEL_AGE")

    def _post_promotion_safety_gate(self, now):
        state = self._state()
        if not state["previous"] or state.get("observation_completed_for") == state["champion"]:
            return False
        bounds = self.store.prediction_issue_bounds(self.key, state["champion"])
        if not bounds:
            return False
        width = self.config.release.minimum_origins * self.config.step_ns
        expected = issue_window(state["model_since"], width, self.config.step_ns, bounds[0])
        previous = self._load_model(state["previous"])
        windows = []
        while expected.start < min(now, state["observation_until"]):
            windows.append(expected)
            expected = range(expected.stop, expected.stop + width, self.config.step_ns)
        if now >= state["observation_until"]:
            # A duration need not be divisible by minimum_origins. Include the
            # full completed period so a short final tail cannot escape review.
            windows.append(issue_window(state["model_since"], state["observation_until"] - state["model_since"],
                                        self.config.step_ns, bounds[0]))
        for expected in windows:
            left = self.store.prediction_rows_by_issue(self.key, state["previous"], expected.start, expected.stop, now)
            right = self.store.prediction_rows_by_issue(self.key, state["champion"], expected.start, expected.stop, now)
            evidence = paired_comparison(left, right, self.config, previous.scale, expected_origins=expected)
            issues = set(evidence["issues"]) & {"CRITICAL_VARIABLE_REGRESSION", "WORST_BUSINESS_GROUP_REGRESSION", "INFERENCE_BUDGET"}
            qualified = evidence["coverage"] == 1. and enough(evidence["champion"], self.config) and enough(evidence["challenger"], self.config)
            if qualified and evidence["champion"]["clean_inputs"] and evidence["challenger"]["clean_inputs"]:
                if evidence["challenger"]["normalized_mse"] > max(evidence["champion"]["normalized_mse"], 1e-12) * self.config.release.rollback_mse_ratio:
                    issues.add("PAIRED_POST_PROMOTION_MSE_REGRESSION")
                if issues:
                    self.store.event(self.key, "POST_PROMOTION_SAFETY_REJECTED", {"model_id": state["champion"],
                        "rollback_target": state["previous"], "issues": sorted(issues), "evidence": evidence}, now)
                    self.rollback("POST_PROMOTION_DEGRADATION")
                    return True
        return False

    def advance(self, now: int):
        lease_seconds = self.config.source.retries * self.config.source.timeout_seconds + 30
        if not self.store.acquire(self.key, self.owner, lease_seconds):
            raise GateError("TASK_ALREADY_RUNNING")
        try:
            return self._advance(now)
        finally:
            self.store.release(self.key, self.owner)

    def _advance(self, now: int):
        state = self._state()
        if state["stop_requested"]:
            self.pump_training()
            self.store.change(self.key, lambda s: s.update(status="STOPPED"))
            return
        if state["now"] is not None and now < state["now"]:
            raise GateError("EVENT_TIME_REWIND")
        self.pump_training()
        self._consume_training(now)
        state = self._state()
        if (state.get("stage") == "INITIAL_APPROVAL" and state.get("approved")
                and state.get("decision_owner", "workflow") == "workflow"):
            try:
                self._load_model(state["challenger"])
                self.store.switch(self.key, None, state["challenger"], now,
                                  {"passed": True, "kind": "INITIAL_OFFLINE_GATE"}, "initial:" + state["challenger"])
            except GateError as error:
                self.store.change(self.key, lambda s: s.update(approved=False, status="AWAITING_APPROVAL"),
                                  "APPROVAL_RECHECK_FAILED", {"code": str(error)}, now)
            state = self._state()
        watermark = now - int(self.config.allowed_lateness_seconds * SECOND)
        start = max(0, watermark - (self.config.search.required_history + max(self.config.search.inputs) + self.config.horizon_steps) * self.config.step_ns - self.config.evaluation_ns)
        labels_synced = False
        try:
            # Label settlement is independent of fresh inputs: a stopped producer
            # must not prevent late truth or corrections from reaching old forecasts.
            if hasattr(self.source, "labels_until"):
                mature = self.source.labels_until(now)
                mature = mature.loc[mature.index <= watermark]
                labels = [(int(t), variable, float(value))
                          for t, row in zip(mature.index.to_numpy(dtype=np.int64), mature.to_numpy())
                          for variable, value in zip(mature.columns, row) if np.isfinite(value)]
                versions = {(target, variable): mature.attrs.get("source_versions_by_time", {}).get(
                    target, "source:" + str(mature.attrs.get("availability_by_time", {}).get(target, target)))
                    for target, variable, _ in labels}
                for target, version in mature.attrs.get("withheld_labels_by_time", {}).items():
                    if target <= watermark:
                        for variable in self.config.variables:
                            labels.append((int(target), variable, None))
                            versions[(int(target), variable)] = version
                self.store.record_tick(self.key, now, None, [], labels, label_versions=versions)
                labels_synced = True
            # Keep ingestion and alignment proportional to the rolling window,
            # not the lifetime of a live arrival journal.
            observed = self.source.read_until(now, start=start)
            if observed.empty:
                raise GateError("DATA_INSUFFICIENT")
            anchor = int(observed.index[0])
            forecast_issue = anchor + ((now - anchor) // self.config.step_ns) * self.config.step_ns
            watermark = anchor + ((watermark - anchor) // self.config.step_ns) * self.config.step_ns
            stable = observed.loc[observed.index <= watermark]
            if stable.empty:
                raise GateError("WAITING_FOR_WATERMARK")
            grid = aligned(stable, self.config, end=watermark)
            forecast_grid = aligned(observed.loc[observed.index <= forecast_issue], self.config, end=forecast_issue)
            # Inference consumes values only. Keep training/label metadata on
            # stable and mature, not on the temporary forecast matrix.
            forecast_grid.attrs = {}
            if not labels_synced:
                # A boolean attestation cannot supply missing arrival/version evidence.
                mature = observed.iloc[0:0]
            if not state["champion"] and not state["challenger"] and not self.store.active_job(self.key):
                self._request_training(stable, watermark, now, "INITIAL_TRAINING")
            forecasts = []
            issue = None
            state = self._state()
            if state["champion"] and (state["last_issue"] is None or forecast_issue > state["last_issue"]):
                filled = causal_fill(forecast_grid, self.config)
                retained = state["previous"] if (state.get("observation_until") is not None and now <= state["observation_until"]
                    and state.get("observation_completed_for") != state["champion"]) else None
                for role, model_id in (("champion", state["champion"]), ("challenger", state["challenger"]), ("retained", retained)):
                    if model_id is None or state["stage"] == "INITIAL_APPROVAL":
                        continue
                    try:
                        model = self._load_model(model_id)
                        raw_tail = forecast_grid.tail(model.candidate.input_steps)
                        x = filled.tail(model.candidate.input_steps).to_numpy(dtype=float)
                        quality = "IMPUTED" if raw_tail.isna().any().any() else "OK"
                        if (raw_tail.isna().to_numpy().mean() > self.config.maximum_missing_fraction
                                or (raw_tail.isna().mean() > self.config.maximum_missing_fraction).any()):
                            raise GateError("INPUT_MISSINGNESS")
                        before = time.perf_counter()
                        prediction = model.predict(x)
                        inference_ms = (time.perf_counter() - before) * 1000
                        if inference_ms > min(self.config.release.maximum_inference_ms, self.config.frequency_seconds * 1000):
                            raise GateError("INFERENCE_DEADLINE")
                    except GateError as error:
                        self.store.event(self.key, "INFERENCE_FAILED", {"model_id": model_id, "code": str(error)}, now)
                        if role == "challenger":
                            self._reject(now, str(error))
                        elif role == "retained":
                            continue
                        elif str(error) in {"ARTIFACT_MISSING", "ARTIFACT_TAMPERED", "MODEL_CONTRACT", "MODEL_RELOAD_MISMATCH", "MODEL_PACKAGE_INVALID", "INVALID_PREDICTION", "INFERENCE_DEADLINE"} and state["previous"]:
                            self.rollback(str(error))
                        elif str(error) in {"ARTIFACT_MISSING", "ARTIFACT_TAMPERED", "MODEL_CONTRACT", "MODEL_RELOAD_MISMATCH", "MODEL_PACKAGE_INVALID", "INVALID_PREDICTION", "INFERENCE_DEADLINE"}:
                            self.store.change(self.key, lambda s: s.update(stop_requested=True, status="SERVING_FAULT"),
                                              "SERVING_STOPPED_FOR_FAULT", {"code": str(error)}, now)
                        continue
                    canary = state["stage"] == "CANARY" and state["challenger"] is not None
                    route = int(digest({"task": self.key, "issue": forecast_issue})[:8], 16) / 2 ** 32 < self.config.release.canary_fraction
                    published = (role == "champion" and not (canary and route)) or (role == "challenger" and canary and route)
                    for h, values in enumerate(prediction, 1):
                        for variable, value in zip(self.config.variables, values):
                            forecasts.append({"model_id": model_id, "issue_ns": forecast_issue, "target_ns": forecast_issue + h * self.config.step_ns, "horizon": h, "variable": variable, "value": float(value), "role": role, "published": published, "inference_ms": inference_ms, "quality": quality})
                active = self._state()["champion"]
                if any(f["model_id"] == active and f["role"] == "champion" for f in forecasts):
                    issue = forecast_issue
                else:
                    # Keep the issue retryable after a failed inference or a just-completed rollback.
                    forecasts = []
                # If a canary failed, valid Champion records remain the served result for this issue.
                if not any(f["published"] for f in forecasts):
                    for f in forecasts:
                        if f["role"] == "champion":
                            f["published"] = True
            self.store.record_tick(self.key, now, issue, forecasts, [])
            self._monitor(now, watermark, mature if isinstance(self.source, ReplaySource) else stable)
            current = self._state()
            if current["status"] == "DEGRADED":
                self.store.change(self.key, lambda s: s.update(status="SERVING" if s["champion"] else "DATA_BLOCKED"), "DATA_RECOVERED", {}, now)
        except GateError as error:
            if labels_synced:
                # Settle historical evidence without training or promotion when
                # current inputs cannot serve. Keep the source fault as the status.
                for model_id in self.store.model_ids_with_predictions(self.key):
                    try:
                        history, _, _ = self._evaluate_mature_windows(model_id, now,
                            track_degradation=model_id == self._state()["champion"],
                            history=self._state()["monitor"])
                        if model_id == self._state()["champion"]:
                            self.store.change(self.key, lambda s: s.update(monitor=history))
                    except GateError:
                        continue
            self.store.change(self.key, lambda s: s.update(status="DEGRADED" if s["champion"] else "DATA_BLOCKED", now=now,
                monitor={**s["monitor"], "count": 0, "triggered": False,
                         "regimes": {name: {**value, "counts": {}} for name, value in s["monitor"].get("regimes", {}).items()}}),
                "DATA_OR_WORKFLOW_BLOCKED", {"code": str(error)}, now)

    def retrain(self) -> str | None:
        state = self._state()
        if state["now"] is None:
            raise GateError("TASK_NOT_STARTED")
        now = state["now"]
        cutoff = now - int(self.config.allowed_lateness_seconds * SECOND)
        frame = self.source.labels_until(now) if isinstance(self.source, ReplaySource) else self.source.read_until(now)
        if frame.empty:
            raise GateError("DATA_INSUFFICIENT")
        cutoff = int(frame.index[0]) + ((cutoff - int(frame.index[0])) // self.config.step_ns) * self.config.step_ns
        return self._request_training(aligned(frame.loc[frame.index <= cutoff], self.config, end=cutoff), cutoff, now, "MANUAL")

    def stop(self):
        state = self._state()
        self.store.change(self.key, lambda s: s.update(stop_requested=True, status="STOPPING"), "STOP_REQUESTED", {}, state["now"] or 0)
        self.pump_training()
        self.store.change(self.key, lambda s: s.update(status="STOPPED"), "STOPPED", {}, state["now"] or 0)

    def resume(self):
        self.store.change(self.key, lambda s: s.update(stop_requested=False, status="SERVING" if s["champion"] else "CREATED"), "RESUMED", {}, self._state()["now"] or 0)

    def close(self):
        # Shutdown only owned workers. Cancel is explicit; interruption keeps the charged attempt recoverable.
        for run_id, (process, owner) in list(self.processes.items()):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            self.store.fail_stage(run_id, owner, "WORKER_INTERRUPTED", self.config, retryable=True)
        self.processes.clear()
        if self.review_process is not None:
            if self.review_process.poll() is None:
                self.review_process.kill()
            self.review_process.wait(timeout=5)
            self.review_process = None
        self._models.clear()
        self.store.release(self.key, self.owner)
