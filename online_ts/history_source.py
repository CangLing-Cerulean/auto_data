from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from .config import GateError
from .data import request_bytes, profile
from .history_service import CONTRACT, _integer, _finite

PAGE_SIZE = 256


class HistorySource:
    """Read-only paginated snapshots of the loopback immutable receipt journal."""
    def __init__(self, config):
        self.config = config

    def _read(self, now, start, labels):
        start = 0 if start is None else start
        if not _integer(now) or not _integer(start) or start > now:
            raise GateError("HISTORY_SERVICE_QUERY")
        url = os.environ.get(self.config.source.url_env, "")
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username
                or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            raise GateError("HISTORY_SERVICE_CONFIGURATION")
        latest, after, upper = {}, 0, None
        while True:
            query = {"asof": now, "start": start, "after_sequence": after, "limit": PAGE_SIZE}
            if upper is not None:
                query["upper_sequence"] = upper
            try:
                data = json.loads(request_bytes(urllib.request.Request(url.rstrip("/") + "/records?" +
                    urllib.parse.urlencode(query)), self.config.source.timeout_seconds))
            except (OSError, ValueError):
                raise GateError("HISTORY_SERVICE_UNAVAILABLE") from None
            try:
                if (type(data) is not dict or data["contract"] != CONTRACT
                        or type(data["asof"]) is not int or data["asof"] != now
                        or type(data["start"]) is not int or data["start"] != start
                        or data["variables"] != list(self.config.variables)
                        or not _integer(data["upper_sequence"]) or data["upper_sequence"] < after
                        or (upper is not None and data["upper_sequence"] != upper)
                        or type(data["records"]) is not list or len(data["records"]) > PAGE_SIZE):
                    raise ValueError()
                upper = data["upper_sequence"]
                previous = after
                for row in data["records"]:
                    if type(row) is not dict:
                        raise ValueError()
                    sequence, event = row["sequence"], row["event_ns"]
                    receipt, mature, values = row["received_ns"], row["mature_ns"], row["values"]
                    if (not all(_integer(v) for v in (sequence, event, receipt, mature))
                            or not previous < sequence <= upper or not start <= event <= receipt <= now
                            or mature < receipt or type(values) is not dict
                            or set(values) != set(self.config.variables)
                            or any(not _finite(v) for v in values.values())):
                        raise ValueError()
                    previous = sequence
                    latest[event] = row
                cursor = data["next_after"]
                if cursor is None:
                    break
                if (not _integer(cursor, 1) or not data["records"] or cursor != previous
                        or not after < cursor < upper or len(data["records"]) != PAGE_SIZE):
                    raise ValueError()
                after = cursor
            except (KeyError, TypeError, ValueError, OverflowError):
                raise GateError("HISTORY_SERVICE_CONTRACT") from None
        # Resolve revisions first: an immature replacement masks the older label.
        withheld = {t: r["sequence"] for t, r in latest.items() if r["mature_ns"] > now} if labels else {}
        if labels:
            latest = {t: r for t, r in latest.items() if r["mature_ns"] <= now}
        frame = pd.DataFrame.from_dict({t: r["values"] for t, r in latest.items()}, orient="index").reindex(columns=self.config.variables).sort_index()
        frame.index = pd.Index(frame.index, name="time", dtype="int64")
        frame.attrs.update(as_of=now, historical_availability="server_receipt_journal",
            revision_visibility="append_only", max_available_at=max(
                (r["mature_ns"] if labels else r["received_ns"] for r in latest.values()), default=now),
            max_revision_at=max((r["received_ns"] for r in latest.values()), default=now),
            availability_by_time={t: r["received_ns"] for t, r in latest.items()},
            label_availability_by_time={t: r["mature_ns"] for t, r in latest.items()},
            source_versions_by_time={t: f"journal:{r['sequence']}" for t, r in latest.items()},
            withheld_labels_by_time={t: f"journal:{sequence}:immature" for t, sequence in withheld.items()})
        return frame

    def read_until(self, now, start=None):
        return self._read(now, start, False)

    def labels_until(self, now, start=None):
        return self._read(now, start, True)

    def discover(self, now):
        return profile(self.read_until(now), self.config)
