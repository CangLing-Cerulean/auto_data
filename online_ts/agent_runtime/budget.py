"""One transactionally accounted total budget; closing calls are not free."""
from ..config import GateError, canonical, digest


CLOSING_CALLS = 2
CLOSING_RESERVATION = 7000
EXPLORATION_OUTPUT = 4096
CLOSING_OUTPUT = 600
EXPLORATION_TOOLS = {"create_experiment", "start_experiment", "run_experiment"}


def is_open(repo):
    return (repo.binding()["policy"] or {}).get("execution_profile", "bounded") == "research_open"


class AgentBudgetError(GateError):
    def __init__(self, reason_code, details=None):
        super().__init__("AGENT_CONTEXT_BUDGET" if reason_code == "CONTEXT_LIMIT" else "AGENT_CALL_BUDGET")
        self.reason_code, self.details = reason_code, details or {}


def usage(repo, db=None):
    if db is None:
        with repo.store.connection() as connection:
            return usage(repo, connection)
    epoch = "agent-epoch-" + str(repo.binding()["epoch"])
    row = db.execute("SELECT COUNT(*),COALESCE(SUM(COALESCE(actual_tokens,reserved_tokens)),0) FROM agent_calls WHERE task_key=? AND run_id=? AND role='agent_controller'",
                     (repo.key, epoch)).fetchone()
    closing = db.execute("SELECT COUNT(*) FROM agent_calls WHERE task_key=? AND run_id=? AND role='agent_controller' AND json_extract(payload,'$.phase')='closing' AND json_extract(payload,'$.incident_id')=?",
                         (repo.key, epoch, repo.state().incident_id)).fetchone()[0]
    return {"calls_used": row[0], "tokens_charged_or_reserved": row[1], "closing_calls_used": closing}


def snapshot(repo):
    policy = repo.binding()["policy"] or {}
    used = usage(repo)
    if is_open(repo):
        return {**used, "execution_profile": "research_open", "max_calls": None, "max_tokens": None,
                "remaining_calls": None, "remaining_tokens": None, "closing_calls_reserved": 0,
                "closing_reservation_per_call": 0}
    result = {**used, "max_calls": policy.get("max_calls", 128), "max_tokens": policy.get("max_llm_tokens", 1_000_000),
              "closing_calls_reserved": max(0, CLOSING_CALLS - used["closing_calls_used"]),
              "closing_reservation_per_call": CLOSING_RESERVATION}
    result["remaining_calls"] = max(0, result["max_calls"] - used["calls_used"])
    result["remaining_tokens"] = max(0, result["max_tokens"] - used["tokens_charged_or_reserved"])
    return result


def check_request(repo, reservation, phase, db=None):
    policy = repo.binding()["policy"] or {}
    used = usage(repo, db)
    if reservation > policy.get("max_call_tokens", 32768) or phase == "closing" and reservation > CLOSING_RESERVATION:
        raise AgentBudgetError("CONTEXT_LIMIT", {"reservation": reservation})
    if is_open(repo):
        if phase == "closing":
            raise AgentBudgetError("OPEN_FORBIDS_FORCED_CLOSING")
        return
    reserved_calls = max(0, CLOSING_CALLS - used["closing_calls_used"]) if phase == "exploration" else 0
    if used["calls_used"] + 1 + reserved_calls > policy.get("max_calls", 128):
        raise AgentBudgetError("CALL_COUNT_LIMIT", used)
    if used["tokens_charged_or_reserved"] + reservation + reserved_calls * CLOSING_RESERVATION > policy.get("max_llm_tokens", 1_000_000):
        raise AgentBudgetError("INSUFFICIENT_TOKEN_RESERVATION", {**used, "requested_reservation": reservation,
                                                               "protected_closing_tokens": reserved_calls * CLOSING_RESERVATION})


def admit_experiment(repo):
    if is_open(repo):
        return
    state = repo.state()
    if getattr(state, "phase", "exploration") == "closing":
        raise AgentBudgetError("CLOSING_ONLY", snapshot(repo))
    budget = snapshot(repo)
    required_calls = max(1, CLOSING_CALLS - budget["closing_calls_used"])
    if budget["remaining_calls"] < required_calls:
        raise AgentBudgetError("CALL_COUNT_LIMIT", budget)
    if budget["remaining_tokens"] < required_calls * CLOSING_RESERVATION:
        raise AgentBudgetError("INSUFFICIENT_TOKEN_RESERVATION", budget)


def record_boundary(repo, now, phase, reservation, reason=None):
    value = {"task_key": repo.key, "phase": phase, "reservation": reservation, "budget": snapshot(repo),
             "reason_code": reason, "decision_version": repo.state().decision_version}
    rid = "budget-" + digest(value)[:24]
    repo.put(rid, "budget_boundary", {"record_id": rid, **value}, now)
    return value
