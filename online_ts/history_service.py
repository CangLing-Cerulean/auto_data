"""Local append-only arrival journal. Receipt time is assigned by the service."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def serve(database, port):
    path = Path(database).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS arrivals (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_ns INTEGER NOT NULL,
                received_ns INTEGER NOT NULL, mature_ns INTEGER NOT NULL, values_json TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON arrivals
                BEGIN SELECT RAISE(ABORT, 'append only'); END;
            CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON arrivals
                BEGIN SELECT RAISE(ABORT, 'append only'); END;
        """)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/records":
                return self.reply(404, {})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError()
                value = json.loads(self.rfile.read(length))
                if set(value) != {"event_ns", "values", "maturity_delay_seconds"}:
                    raise ValueError()
                now = time.time_ns()
                delay = value["maturity_delay_seconds"]
                if (type(value["event_ns"]) is not int or value["event_ns"] > now
                        or type(delay) not in {int, float} or not math.isfinite(delay) or delay < 0
                        or not isinstance(value["values"], dict) or not value["values"]
                        or any(not isinstance(k, str) or type(v) not in {int, float} or not math.isfinite(v)
                               for k, v in value["values"].items())):
                    raise ValueError()
                with sqlite3.connect(path) as db:
                    cursor = db.execute("INSERT INTO arrivals(event_ns,received_ns,mature_ns,values_json) VALUES(?,?,?,?)",
                        (value["event_ns"], now, now + int(delay * 1e9), json.dumps(value["values"], sort_keys=True)))
                    sequence = cursor.lastrowid
                self.reply(201, {"sequence": sequence, "received_ns": now})
            except (ValueError, TypeError, KeyError):
                self.reply(400, {"error": "RECORD_SCHEMA"})

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path != "/records":
                return self.reply(404, {})
            try:
                query = parse_qs(parsed.query)
                asof = int(query["asof"][0])
                start = int(query.get("start", [0])[0])
                with sqlite3.connect(path) as db:
                    rows = db.execute("SELECT sequence,event_ns,received_ns,mature_ns,values_json FROM arrivals WHERE event_ns>=? AND event_ns<=? AND received_ns<=? ORDER BY sequence",
                                      (start, asof, asof)).fetchall()
                self.reply(200, {"contract": "server-receipt-journal-v1", "asof": asof,
                    "records": [{"sequence": r[0], "event_ns": r[1], "received_ns": r[2],
                                 "mature_ns": r[3], "values": json.loads(r[4])} for r in rows]})
            except (ValueError, KeyError):
                self.reply(400, {"error": "QUERY_SCHEMA"})

    # This controlled service is deliberately local. Remote authentication and
    # field-specific maturity authority require separate deployment validation.
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--port", type=int, default=18761)
    args = parser.parse_args()
    serve(args.database, args.port)
