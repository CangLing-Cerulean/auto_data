"""Synthetic v2 journal tests; no private configuration or research data."""
import concurrent.futures
import copy
import http.client
import json
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.parse
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from online_ts.config import GateError
from online_ts.history_service import CONTRACT, MAX_INTEGER, create_server
from online_ts.history_source import HistorySource


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.path = Path(self.temp.name) / "journal.sqlite"
        self.server = create_server(self.path, variables=("x",))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port
        self.config = SimpleNamespace(variables=("x",), source=SimpleNamespace(
            url_env="SYNTHETIC_JOURNAL_URL", timeout_seconds=2))
        self.env = patch.dict("os.environ", {"SYNTHETIC_JOURNAL_URL": f"http://127.0.0.1:{self.port}"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, method, path, value=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            body = json.dumps(value) if value is not None else None
            connection.request(method, path, body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def record(self, key="one", event=1, x=2, delay=0):
        return {"idempotency_key": key, "event_ns": event,
                "values": {"x": x}, "maturity_delay_seconds": delay}

    def query(self, **kwargs):
        return self.request("GET", "/records?" + urllib.parse.urlencode(kwargs))

    def test_idempotency_conflict_and_append_only(self):
        payload = self.record()
        status, first = self.request("POST", "/records", payload)
        self.assertEqual(status, 201)
        status, repeated = self.request("POST", "/records", payload)
        self.assertEqual((status, repeated), (200, first))
        self.assertEqual(self.request("POST", "/records", self.record(x=3))[0], 409)
        with closing(sqlite3.connect(self.path)) as db, db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM arrivals").fetchone()[0], 1)
            for sql in ("DELETE FROM arrivals", "UPDATE arrivals SET event_ns=2"):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)
        self.assertEqual(self.request("GET", "/health")[1]["contract"], CONTRACT)

    def test_concurrent_same_key_has_one_arrival(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(lambda _: self.request("POST", "/records", self.record()), range(6)))
        self.assertEqual(sum(s == 201 for s, _ in responses), 1)
        self.assertTrue(all(body == responses[0][1] for _, body in responses))

    def test_retry_keeps_arrival_even_when_clock_moves_back(self):
        event = time.time_ns() - 1000000
        payload = self.record(event=event)
        _, first = self.request("POST", "/records", payload)
        with patch("online_ts.history_service.time.time_ns", return_value=event - 1):
            self.assertEqual(self.request("POST", "/records", payload), (200, first))

    def test_revisions_maturity_start_and_availability(self):
        _, old = self.request("POST", "/records", self.record(event=10))
        before = time.time_ns()
        _, new = self.request("POST", "/records", self.record("new", event=10, x=8, delay=60))
        self.request("POST", "/records", self.record("other", event=20, x=4))
        source, now = HistorySource(self.config), time.time_ns()
        self.assertEqual(source.read_until(before).loc[10, "x"], 2)
        self.assertEqual(source.labels_until(before).loc[10, "x"], 2)
        self.assertEqual(source.read_until(now).loc[10, "x"], 8)
        self.assertNotIn(10, source.labels_until(now).index)
        bounded = source.read_until(now, start=20)
        self.assertEqual(bounded.index.tolist(), [20])
        frame = source.read_until(now)
        self.assertEqual(frame.attrs["availability_by_time"][10], new["received_ns"])
        self.assertEqual(frame.attrs["label_availability_by_time"][10], new["mature_ns"])
        self.assertLessEqual(old["received_ns"], before)

    def test_pagination_freezes_upper_and_source_uses_start(self):
        for i in range(5):
            self.request("POST", "/records", self.record(str(i), event=i + 1))
        asof = time.time_ns()
        status, first = self.query(asof=asof, start=2, limit=2)
        self.assertEqual(status, 200)
        self.assertEqual([r["sequence"] for r in first["records"]], [2, 3])
        self.assertEqual(first["next_after"], 3)
        self.request("POST", "/records", self.record("later", event=1))
        _, second = self.query(asof=asof, start=2, limit=2,
            after_sequence=first["next_after"], upper_sequence=first["upper_sequence"])
        self.assertEqual([r["sequence"] for r in second["records"]], [4, 5])
        self.assertIsNone(second["next_after"])
        with patch("online_ts.history_source.PAGE_SIZE", 2):
            frame = HistorySource(self.config).read_until(asof, start=2)
        self.assertEqual(frame.index.tolist(), [2, 3, 4, 5])

    def test_invalid_records_and_queries(self):
        invalid = [self.record(event=-1), self.record(event=True), self.record(event=MAX_INTEGER + 1),
            self.record(event=time.time_ns() + 10**12), self.record(x=True), self.record(x=None),
            self.record(x=float("nan")), self.record(x=float("inf")), self.record(x=10**400),
            self.record(delay=float("inf")), self.record(delay=True), self.record(delay=-1),
            self.record(delay=MAX_INTEGER), self.record(key=""), self.record(key="a" * 257)]
        mismatch = self.record()
        mismatch["values"] = {"y": 2}
        invalid.extend([mismatch, [], None])
        for value in invalid:
            with self.subTest(value=repr(value)[:90]):
                self.assertEqual(self.request("POST", "/records", value)[0], 400)
        for query in ({"asof": -1}, {"asof": MAX_INTEGER + 1}, {"asof": 1, "limit": 1001},
                      {"asof": 1, "limit": 0}, {"asof": 1, "start": 2},
                      {"asof": 1, "after_sequence": 2, "upper_sequence": 1}):
            self.assertEqual(self.query(**query)[0], 400)
        self.assertEqual(self.request("GET", "/records?asof=1&asof=2")[0], 400)

    def test_body_bounds_and_timeout(self):
        for headers, body in (("", b""), ("Content-Length: 65537\r\n", b""),
                ("Content-Length: -1\r\n", b""),
                ("Content-Length: 2\r\nContent-Length: 2\r\n", b"{}"),
                ("Transfer-Encoding: chunked\r\nContent-Length: 2\r\n", b"{}")):
            with socket.create_connection(("127.0.0.1", self.port), timeout=2) as client:
                client.sendall(("POST /records HTTP/1.0\r\n" + headers + "\r\n").encode() + body)
                self.assertIn(b" 400 ", client.recv(4096))
        with patch("online_ts.history_service.SOCKET_TIMEOUT_SECONDS", 0.05):
            with socket.create_connection(("127.0.0.1", self.port), timeout=2) as client:
                client.sendall(b"POST /records HTTP/1.0\r\nContent-Length: 20\r\n\r\n{")
                self.assertIn(b" 408 ", client.recv(4096))

    def test_v1_and_variable_changes_rejected(self):
        old_path = Path(self.temp.name) / "old.sqlite"
        with closing(sqlite3.connect(old_path)) as db, db:
            db.execute("CREATE TABLE arrivals(sequence INTEGER PRIMARY KEY,event_ns INTEGER)")
        with self.assertRaisesRegex(ValueError, "V2_REQUIRED"):
            create_server(old_path, variables=("x",))
        with self.assertRaisesRegex(ValueError, "V2_REQUIRED"):
            create_server(self.path, variables=("y",))
        with create_server(self.path, variables=("x",)):
            pass

    def test_source_rejects_malformed_contract(self):
        base = {"contract": CONTRACT, "variables": ["x"], "asof": 100, "start": 0,
            "upper_sequence": 1, "next_after": None,
            "records": [{"sequence": 1, "event_ns": 1, "received_ns": 2,
                         "mature_ns": 3, "values": {"x": 1}}]}
        invalid = []
        for field, value in (("contract", "server-receipt-journal-v1"), ("asof", 101),
                ("start", 1), ("variables", ["y"]), ("next_after", 0),
                ("next_after", 1), ("upper_sequence", True), ("records", {})):
            item = copy.deepcopy(base)
            item[field] = value
            invalid.append(item)
        for field, value in (("sequence", True), ("event_ns", -1), ("event_ns", 101),
                ("received_ns", 101), ("mature_ns", MAX_INTEGER + 1),
                ("values", {"x": float("nan")}), ("values", {"x": True}), ("values", {"y": 1})):
            item = copy.deepcopy(base)
            item["records"][0][field] = value
            invalid.append(item)
        duplicate = copy.deepcopy(base)
        duplicate["records"] *= 2
        invalid.extend([duplicate, [], {}])
        for item in invalid:
            with self.subTest(item=item), patch("online_ts.history_source.request_bytes", return_value=json.dumps(item).encode()):
                with self.assertRaisesRegex(GateError, "CONTRACT"):
                    HistorySource(self.config).read_until(100)

    def test_source_rejects_snapshot_change_and_cursor_regression(self):
        page = {"contract": CONTRACT, "variables": ["x"], "asof": 100, "start": 0,
            "upper_sequence": 3, "next_after": 1,
            "records": [{"sequence": 1, "event_ns": 1, "received_ns": 2,
                         "mature_ns": 3, "values": {"x": 1}}]}
        for changed in (dict(page, upper_sequence=4), page):
            with patch("online_ts.history_source.PAGE_SIZE", 1), patch(
                    "online_ts.history_source.request_bytes", side_effect=[json.dumps(page).encode(), json.dumps(changed).encode()]):
                with self.assertRaisesRegex(GateError, "CONTRACT"):
                    HistorySource(self.config).read_until(100)


if __name__ == "__main__":
    unittest.main()
