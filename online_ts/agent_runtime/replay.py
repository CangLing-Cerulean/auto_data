"""Independent accelerated serving loop; observation snapshots never end incidents."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import threading
import time
import uuid

from ..config import GateError, SECOND
from ..engine import Engine
from .controller import AgentController
from .event_loop import EventLoop
from .llm import DeepSeekModel
from .state import Repository
from .tools.gateway import ToolGateway


DEFAULT_REPLAY_RATE = 5.0
OBSERVATION_SECONDS = 1800


class _ReplayToolGateway(ToolGateway):
    def __init__(self, engine, repo, execution_lock, tool_requested):
        super().__init__(engine, repo)
        self.execution_lock, self.tool_requested = execution_lock, tool_requested

    def execute(self, record):
        # Give an already selected tool the next local execution slot. The
        # normal task lease still guards against other processes and owners.
        self.tool_requested.set()
        try:
            with self.execution_lock:
                return super().execute(record)
        finally:
            self.tool_requested.clear()


class ReplayRunner:
    def __init__(self, engine, repo, *, model=None, rate=DEFAULT_REPLAY_RATE,
                 monotonic=time.monotonic, wall_ns=time.time_ns, on_decision=None, on_observation=None):
        if engine.config.mode != "replay" or not 0 < rate <= 100:
            raise GateError("RESEARCH_REPLAY_CONFIGURATION")
        self.engine, self.repo, self.rate = engine, repo, rate
        self.monotonic, self.wall_ns = monotonic, wall_ns
        self.on_decision, self.on_observation = on_decision, on_observation
        self.owner = "replay-" + uuid.uuid4().hex
        self.lease = "research-replay:" + repo.key
        if not repo.store.acquire(self.lease, self.owner, 30):
            raise GateError("RESEARCH_RUNNER_ALREADY_ACTIVE")
        self.control = None
        self.executor = None
        self.closed = False
        try:
            self._initialize(model)
        except BaseException:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
            if self.control is not None:
                self.control.close()
            repo.store.release(self.lease, self.owner)
            raise

    def _initialize(self, model):
        engine, repo = self.engine, self.repo
        self.control = Engine(engine.runtime, engine.key)
        self.control_repo = Repository(self.control.store, self.control.key)
        self.execution_lock = threading.Lock()
        self.tool_requested = threading.Event()
        gateway = _ReplayToolGateway(self.control, self.control_repo, self.execution_lock, self.tool_requested)
        from .observers import ObservationTeam
        self.observation_team = ObservationTeam(ToolGateway(engine, repo), notify=True)
        self.loop = EventLoop(AgentController(self.control_repo, gateway,
                             model or DeepSeekModel(self.control_repo, engine.config)),
                             inspection_seconds=(repo.binding()["policy"] or {}).get("inspection_seconds", 3600),
                             observation_team=self.observation_team)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="open-agent")
        self.future = None
        self.started = self.monotonic()
        self.last_pulse = self.started
        self.last_lease = self.started
        self.processed = engine._state()["now"]
        if self.processed is None:
            self.processed = engine.source.bounds[0] + (engine.config.search.required_history - 1) * engine.config.step_ns
        self.origin = self.processed
        try:
            checkpoint = repo.get("replay-clock")
            self.elapsed_before = checkpoint["elapsed_wall_seconds"]
            self.next_observation = checkpoint["next_observation_seconds"]
        except GateError as error:
            if str(error) != "AGENT_RECORD_NOT_FOUND":
                raise
            self.elapsed_before, self.next_observation = 0., OBSERVATION_SECONDS
        self.terminal = None
        self.checkpoint()

    def checkpoint(self):
        elapsed = self.elapsed_before + max(0., self.monotonic() - self.started)
        due = self.origin + int(max(0., self.monotonic() - self.started) * self.rate * SECOND)
        self.repo.put("replay-clock", "replay_clock", {"event_time": self.processed, "wall_ns": self.wall_ns(),
            "rate": self.rate, "elapsed_wall_seconds": elapsed, "next_observation_seconds": self.next_observation,
            "processing_lag_event_seconds": max(0, due - self.processed) / SECOND,
            "resume_policy": "resume_processed_position_without_catching_up_downtime"}, self.processed, replace=True)

    def pulse(self):
        """Perform one nonblocking supervisor pulse. Slow inference never gates advance."""
        if self.closed:
            raise GateError("RESEARCH_RUNNER_CLOSED")
        if self.terminal:
            return self.terminal
        tick = self.monotonic()
        if tick - self.last_lease >= 10:
            if not self.repo.store.acquire(self.lease, self.owner, 30):
                raise GateError("RESEARCH_RUNNER_LEASE_LOST")
            self.last_lease = tick
        state = self.engine._state()
        if state["stop_requested"]:
            self.terminal = "operator_stop"
            return self.terminal
        target = self.origin + int(max(0, tick - self.started) * self.rate * SECOND)
        last = self.engine.source.bounds[1]
        following = self.processed + self.engine.config.step_ns
        if self.execution_lock.acquire(blocking=False):
            try:
                if not self.tool_requested.is_set():
                    if following <= min(target, last):
                        try:
                            self.engine.advance(following)
                            self.processed = following
                            self.observation_team.refresh(self.processed)
                        except GateError as error:
                            if str(error) != "TASK_ALREADY_RUNNING":
                                raise
                    else:
                        self.engine.pump_training()
            finally:
                self.execution_lock.release()
        if self.future is not None and self.future.done():
            record = self.future.result()
            self.future = None
            if record and self.on_decision:
                self.on_decision(record)
        exhausted = self.processed + self.engine.config.step_ns > last and target >= last
        if not exhausted and self.future is None and self.repo.binding()["owner"] == "agent" and self.repo.state().status != "faulted":
            self.future = self.executor.submit(self.loop.tick, self.processed)
        elapsed = self.elapsed_before + tick - self.started
        if elapsed >= self.next_observation:
            snapshot = {"event_time": self.processed, "wall_ns": self.wall_ns(), "elapsed_wall_seconds": elapsed,
                        "state": asdict(self.repo.state()), "serving": self.engine._state(),
                        "observation_only": True, "forces_finalize": False}
            self.repo.put("observation-" + str(self.next_observation), "research_observation", snapshot, self.processed)
            self.next_observation += OBSERVATION_SECONDS
            if self.on_observation:
                self.on_observation(snapshot)
        if tick - self.last_pulse >= 1:
            self.checkpoint()
            self.last_pulse = tick
        if exhausted:
            self.terminal = "source_exhausted"
            self.repo.put("source-exhausted", "research_interruption", {"reason": self.terminal,
                          "event_time": self.processed, "wall_ns": self.wall_ns(), "agent_completed": False}, self.processed)
        return self.terminal

    def run(self):
        try:
            while not self.pulse():
                time.sleep(.01)
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            # In-flight HTTP is bounded by the transport deadline. Keep its audit
            # and database alive until it returns; do not launch another action.
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.checkpoint()
        finally:
            try:
                self.control.close()
            finally:
                self.repo.store.release(self.lease, self.owner)
