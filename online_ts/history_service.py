"""Append-only v2 receipt journal for loopback HTTP producers."""
from __future__ import annotations

import argparse
import json
import math
import socket
import sqlite3
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

CONTRACT = "server-receipt-journal-v2"
MAX_INTEGER = (1 << 63) - 1
MAX_BODY = 64 * 1024
SOCKET_TIMEOUT_SECONDS = 5


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= MAX_INTEGER


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value):
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def create_server(database, port=0, *, variables):
    """Create without starting; callers own serve/shutdown/close."""
    if isinstance(variables, str):
        raise ValueError("JOURNAL_VARIABLES")
    variables = tuple(variables)
    if (not variables or any(type(v) is not str or not v.strip() for v in variables)
            or len(set(variables)) != len(variables)):
        raise ValueError("JOURNAL_VARIABLES")
    path = Path(database).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        if tables:
            if tables != {"journal_metadata", "arrivals"}:
                raise ValueError("JOURNAL_SCHEMA_V2_REQUIRED")
            metadata = db.execute("SELECT contract,variables_json FROM journal_metadata").fetchall()
            columns = [r[1] for r in db.execute("PRAGMA table_info(arrivals)")]
            if (metadata != [(CONTRACT, _json(variables))] or columns != ["sequence", "idempotency_key",
                    "payload_json", "event_ns", "received_ns", "mature_ns", "values_json"]):
                raise ValueError("JOURNAL_SCHEMA_V2_REQUIRED")
        else:
            db.executescript("""
                CREATE TABLE journal_metadata(contract TEXT NOT NULL, variables_json TEXT NOT NULL);
                CREATE TABLE arrivals (sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT NOT NULL UNIQUE, payload_json TEXT NOT NULL,
                    event_ns INTEGER NOT NULL, received_ns INTEGER NOT NULL,
                    mature_ns INTEGER NOT NULL, values_json TEXT NOT NULL);
                CREATE INDEX arrivals_visibility ON arrivals(received_ns,sequence);
                CREATE INDEX arrivals_event ON arrivals(event_ns,sequence);
                CREATE TRIGGER no_update BEFORE UPDATE ON arrivals
                    BEGIN SELECT RAISE(ABORT, 'append only'); END;
                CREATE TRIGGER no_delete BEFORE DELETE ON arrivals
                    BEGIN SELECT RAISE(ABORT, 'append only'); END;
                CREATE TRIGGER metadata_no_update BEFORE UPDATE ON journal_metadata
                    BEGIN SELECT RAISE(ABORT, 'immutable metadata'); END;
                CREATE TRIGGER metadata_no_delete BEFORE DELETE ON journal_metadata
                    BEGIN SELECT RAISE(ABORT, 'immutable metadata'); END;
            """)
            db.execute("INSERT INTO journal_metadata VALUES(?,?)", (CONTRACT, _json(variables)))

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(SOCKET_TIMEOUT_SECONDS)

        def log_message(self, *args):
            pass

        def reply(self, status, value):
            body = _json(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/records":
                return self.reply(404, {})
            try:
                lengths = self.headers.get_all("Content-Length", [])
                if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                        or self.headers.get("Transfer-Encoding") is not None):
                    raise ValueError()
                length = int(lengths[0])
                if not 0 < length <= MAX_BODY:
                    raise ValueError()
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError()
                value, now = json.loads(raw), time.time_ns()
                if type(value) is not dict or set(value) != {"idempotency_key", "event_ns", "values", "maturity_delay_seconds"}:
                    raise ValueError()
                key, delay, values = value["idempotency_key"], value["maturity_delay_seconds"], value["values"]
                if (type(key) is not str or not key.strip() or len(key) > 256
                        or not _integer(value["event_ns"])
                        or not _finite(delay) or delay < 0
                        or type(values) is not dict or set(values) != set(variables)
                        or any(not _finite(v) for v in values.values())):
                    raise ValueError()
                payload = _json(value)
            except (ValueError, TypeError, OverflowError):
                self.close_connection = True
                return self.reply(400, {"error": "RECORD_SCHEMA"})
            except (socket.timeout, TimeoutError):
                self.close_connection = True
                return self.reply(408, {"error": "BODY_TIMEOUT"})
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                previous = db.execute("SELECT sequence,received_ns,mature_ns,payload_json FROM arrivals WHERE idempotency_key=?", (key,)).fetchone()
                if previous:
                    if previous[3] != payload:
                        return self.reply(409, {"error": "IDEMPOTENCY_CONFLICT"})
                    sequence, now, mature = previous[:3]
                    status = 200
                else:
                    if value["event_ns"] > now or delay > (MAX_INTEGER - now) / 1e9:
                        return self.reply(400, {"error": "RECORD_SCHEMA"})
                    mature = now + int(delay * 1e9)
                    if not _integer(mature):
                        return self.reply(400, {"error": "RECORD_SCHEMA"})
                    cursor = db.execute("INSERT INTO arrivals(idempotency_key,payload_json,event_ns,received_ns,mature_ns,values_json) VALUES(?,?,?,?,?,?)",
                        (key, payload, value["event_ns"], now, mature, _json(values)))
                    sequence, status = cursor.lastrowid, 201
            self.reply(status, {"sequence": sequence, "received_ns": now, "mature_ns": mature})

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/health" and not parsed.query:
                return self.reply(200, {"contract": CONTRACT, "variables": variables, "status": "ok"})
            if parsed.path != "/records":
                return self.reply(404, {})
            try:
                query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                if (set(query) - {"asof", "start", "after_sequence", "limit", "upper_sequence"}
                        or "asof" not in query or any(len(v) != 1 for v in query.values())):
                    raise ValueError()
                numbers = {}
                for k, v in query.items():
                    if not v[0].isascii() or not v[0].isdigit():
                        raise ValueError()
                    numbers[k] = int(v[0])
                asof, start = numbers["asof"], numbers.get("start", 0)
                after, limit = numbers.get("after_sequence", 0), numbers.get("limit", 1000)
                upper = numbers.get("upper_sequence")
                if (any(not _integer(v) for v in numbers.values()) or asof > time.time_ns()
                        or start > asof or not 1 <= limit <= 1000 or (upper is not None and after > upper)):
                    raise ValueError()
            except (ValueError, OverflowError):
                return self.reply(400, {"error": "QUERY_SCHEMA"})
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("BEGIN")
                if upper is None:
                    upper = db.execute("SELECT COALESCE(MAX(sequence),0) FROM arrivals WHERE received_ns<=?", (asof,)).fetchone()[0]
                if after > upper:
                    return self.reply(400, {"error": "QUERY_SCHEMA"})
                rows = db.execute("SELECT sequence,event_ns,received_ns,mature_ns,values_json FROM arrivals WHERE sequence>? AND sequence<=? AND event_ns>=? AND event_ns<=? AND received_ns<=? ORDER BY sequence LIMIT ?",
                    (after, upper, start, asof, asof, limit + 1)).fetchall()
            more, rows = len(rows) > limit, rows[:limit]
            self.reply(200, {"contract": CONTRACT, "variables": variables, "asof": asof, "start": start,
                "upper_sequence": upper, "next_after": rows[-1][0] if more else None,
                "records": [{"sequence": r[0], "event_ns": r[1], "received_ns": r[2],
                    "mature_ns": r[3], "values": json.loads(r[4])} for r in rows]})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(database, port=18761, *, variables):
    with create_server(database, port, variables=variables) as server:
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--port", type=int, default=18761)
    parser.add_argument("--variables", nargs="+", required=True)
    args = parser.parse_args()
    serve(args.database, args.port, variables=args.variables)
