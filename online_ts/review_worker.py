"""Optional management-plane explanation worker; never drives training or release."""
from __future__ import annotations

import sys
import uuid

from .agent import explain_card
from .store import Store
from .reviews import list_cards


def run(database, key):
    store = Store(database)
    config, state = store.task(key)
    if state["stop_requested"] or config.planner.kind != "external":
        return
    owner = uuid.uuid4().hex
    resource = "model-review:" + key
    lease = config.planner.timeout_seconds * 2 + 30
    if not store.acquire_resource(resource, owner, key, "management", "model-review", lease):
        return
    try:
        # One bounded call per process; no unbounded management loop.
        pending = next((c for c in list_cards(store, key)
                        if c["stage"] != "final_test" and (c["llm_status"] == "queued" or
                            (config.collaboration.automatic_reviews and c["llm_status"] == "pending"))), None)
        if pending is not None:
            from .reviews import queue_explanation
            queue_explanation(store, key, pending["review_id"])
            explain_card(store, config, key, pending["review_id"])
    finally:
        store.release_resource(resource, owner)


if __name__ == "__main__":
    run(*sys.argv[1:])
