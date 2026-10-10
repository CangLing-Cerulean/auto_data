from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from .config import GateError
from .data import request_bytes, profile


class HistorySource:
    """Read-only connector to the local service's immutable receipt journal."""
    def __init__(self, config):
        self.config = config

    def _read(self, now, start, labels):
        url = os.environ.get(self.config.source.url_env, "")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise GateError("HISTORY_SERVICE_CONFIGURATION")
        query = urllib.parse.urlencode({"asof": now, "start": start or 0})
        try:
            data = json.loads(request_bytes(urllib.request.Request(url.rstrip("/") + "/records?" + query), self.config.source.timeout_seconds))
        except (OSError, ValueError):
            raise GateError("HISTORY_SERVICE_UNAVAILABLE") from None
        if data.get("contract") != "server-receipt-journal-v1" or data.get("asof") != now:
            raise GateError("HISTORY_SERVICE_CONTRACT")
        latest = {}
        for row in data["records"]:
            event, receipt, mature = row["event_ns"], row["received_ns"], row["mature_ns"]
            if not event <= receipt <= now or mature < receipt or set(row["values"]) != set(self.config.variables):
                raise GateError("HISTORY_SERVICE_CONTRACT")
            if not labels or mature <= now:
                latest[event] = row
        frame = pd.DataFrame.from_dict({t: r["values"] for t, r in latest.items()}, orient="index").reindex(columns=self.config.variables).sort_index()
        frame.index = pd.Index(frame.index, name="time", dtype="int64")
        frame.attrs.update(as_of=now, historical_availability="server_receipt_journal",
            revision_visibility="append_only", max_available_at=max((r["received_ns"] for r in latest.values()), default=now),
            max_revision_at=max((r["received_ns"] for r in latest.values()), default=now),
            availability_by_time={t: r["received_ns"] for t, r in latest.items()},
            label_availability_by_time={t: r["mature_ns"] for t, r in latest.items()})
        return frame

    def read_until(self, now, start=None):
        return self._read(now, start, False)

    def labels_until(self, now, start=None):
        return self._read(now, start, True)

    def discover(self, now):
        return profile(self.read_until(now), self.config)
