"""Opt-in CLI. The existing `agent` CLI and TaskConfig remain unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from ..config import GateError, SECOND
from ..engine import Engine
from .controller import AgentController
from .event_loop import EventLoop
from .llm import DeepSeekModel
from .policies import AutonomyPolicy, configure
from .state import Repository
from .tools.gateway import ToolGateway


def run(engine, repo, *, ticks=None, tick_seconds=1):
    from .budget import is_open
    if is_open(repo):
        if ticks is not None:
            raise GateError("OPEN_FORBIDS_TICK_LIMIT")
        from .replay import ReplayRunner
        ReplayRunner(engine, repo, on_decision=lambda r: print(json.dumps(r, ensure_ascii=False), flush=True)).run()
        return
    # A separate Engine facade for the controller shares durable Store/Artifacts,
    # not subprocess handles. Main-thread serving never awaits LLM I/O.
    control_engine = Engine(engine.runtime, engine.key)
    controller = AgentController(repo, ToolGateway(control_engine, repo), DeepSeekModel(repo, engine.config))
    inspection = (repo.binding()["policy"] or {}).get("inspection_seconds", 3600)
    from .observers import ObservationTeam
    observation_team = ObservationTeam(ToolGateway(engine, repo), notify=True)
    loop = EventLoop(controller, inspection_seconds=inspection, observation_team=observation_team)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-decision")
    future = None
    count = 0
    try:
        while ticks is None or count < ticks:
            state = engine._state()
            if state["stop_requested"]:
                break
            if engine.config.mode == "live":
                now = time.time_ns()
            else:
                first, last = engine.source.bounds
                initial = first + (engine.config.search.required_history - 1) * engine.config.step_ns
                now = min(last, (state["now"] + engine.config.step_ns) if state["now"] is not None else initial)
            try:
                engine.advance(now)
                observation_team.refresh(now)
            except GateError as error:
                if str(error) != "TASK_ALREADY_RUNNING":
                    raise
            if future is not None and future.done():
                result = future.result()
                if result:
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                future = None
            if future is None and repo.binding()["owner"] == "agent":
                future = executor.submit(loop.tick, now)
            count += 1
            time.sleep(tick_seconds)
    finally:
        # Cancellation flag in the existing network transport observes stop/epoch.
        # No switch of decision owner is inferred from runner shutdown.
        executor.shutdown(wait=True, cancel_futures=True)
        control_engine.close()


def main():
    parser = argparse.ArgumentParser(description="Independent LLM-led forecasting operations controller")
    parser.add_argument("--runtime", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("configure")
    setup.add_argument("task")
    setup.add_argument("--controller-mode", choices=["workflow", "agent"], default="workflow")
    setup.add_argument("--policy")
    setup.add_argument("--goal")
    worker = sub.add_parser("run")
    worker.add_argument("task")
    worker.add_argument("--ticks", type=int)
    worker.add_argument("--tick-seconds", type=float, default=1)
    for name in ("status", "decisions", "resume"):
        sub.add_parser(name).add_argument("task")
    approval = sub.add_parser("approve-release")
    approval.add_argument("task")
    approval.add_argument("experiment_id")
    args = parser.parse_args()
    engine = Engine(args.runtime, args.task)
    repo = Repository(engine.store, engine.key)
    try:
        if args.command == "configure":
            policy = json.loads(Path(args.policy).read_text(encoding="utf-8-sig")) if args.policy else None
            configure(repo, args.controller_mode, policy=policy, goal=args.goal)
            print(json.dumps(repo.binding(), ensure_ascii=False, indent=2))
        elif args.command == "run":
            if args.tick_seconds <= 0 or args.ticks is not None and args.ticks < 1:
                raise GateError("AGENT_RUN_LIMIT")
            run(engine, repo, ticks=args.ticks, tick_seconds=args.tick_seconds)
        elif args.command == "decisions":
            print(json.dumps(repo.records("decision", 100), ensure_ascii=False, indent=2))
        elif args.command == "approve-release":
            from .policies import grant_release
            print(json.dumps(grant_release(repo, args.experiment_id), ensure_ascii=False, indent=2))
        elif args.command == "resume":
            from ..copilot import current_actor
            if "experiment" not in engine.config.collaboration.permissions.get(current_actor(), ()):
                raise GateError("HUMAN_PERMISSION_DENIED")
            state = repo.state()
            if state.status != "faulted":
                raise GateError("AGENT_NOT_FAULTED")
            from dataclasses import asdict
            from .state import Event
            identity = "resume-" + str(time.time_ns())
            resumed_at = engine._state()["now"]
            state.status = "investigating"
            state.pending_event = asdict(Event(identity, "HUMAN_RESUMED", resumed_at,
                                              {"actor": current_actor(), "resume_id": identity}))
            with repo.store.transaction() as db:
                repo.save_state(state, state.decision_version, db=db)
                repo.put(identity, "human_resume", {"actor": current_actor(), "wall_ns": time.time_ns(),
                         "resume_id": identity, "event_time": resumed_at}, resumed_at, db=db)
        else:
            from dataclasses import asdict
            print(json.dumps({"binding": repo.binding(), "state": asdict(repo.state())}, ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
