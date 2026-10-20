"""Synthetic HTTP -> training worker -> forecast/monitor -> restart checks."""
import json
import math
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from online_ts.config import TaskConfig, SECOND
from online_ts.copilot import current_actor
from online_ts.engine import Engine
from online_ts.history_service import create_server
from online_ts.platform import streams


def config():
    return TaskConfig.from_dict({
        "task_id": "synthetic_stream", "variables": ["x"], "mode": "live",
        "source": {"kind": "history_http", "url_env": "PLATFORM_TEST_SOURCE",
                   "timeout_seconds": 2, "retries": 1},
        "frequency_seconds": 1, "horizon_steps": 2, "allowed_lateness_seconds": 0,
        "search": {"input_steps": 8, "history_steps": 64, "validation_steps": 12,
                   "test_steps": 12, "seasonal_period": 1,
                   "model_families": ["last_value"], "max_train_samples": 64},
        "budget": {"max_candidates": 1, "max_candidate_seconds": 15,
                   "max_training_seconds": 60, "max_memory_mb": 1024, "max_attempts": 1},
        "monitoring": {"evaluation_seconds": 4, "check_seconds": 1,
                       "minimum_new_steps": 4, "cooldown_seconds": 0,
                       "max_model_age_seconds": 1000000},
        "release": {"minimum_origins": 3, "shadow_seconds": 4, "canary_seconds": 4,
                    "post_promotion_seconds": 4, "maximum_observation_seconds": 120},
        "collaboration": {"permissions": {current_actor(): ["release"]}},
    })


class StreamIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp.name)
        self.server = create_server(self.root / "journal.db", variables=["x"])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.env = patch.dict("os.environ", {"PLATFORM_TEST_SOURCE": self.url})
        self.env.start()
        self.base = (time.time_ns() // SECOND - 1000) * SECOND
        self.clock = patch("online_ts.history_service.time.time_ns", return_value=self.base)
        self.now = self.clock.start()
        self.engine = Engine.create(self.root / "runtime", config())

    def tearDown(self):
        self.engine.close()
        self.clock.stop()
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def append(self, step, *, delay=0, key=None, value=None, received_step=None):
        stamp = self.base + step * SECOND
        self.now.return_value = self.base + (step if received_step is None else received_step) * SECOND
        body = {"idempotency_key": key or f"row-{step}", "event_ns": stamp,
                "values": {"x": value if value is not None else 10 + .03 * step + .1 * math.sin(step / 10)},
                "maturity_delay_seconds": delay}
        request = urllib.request.Request(self.url + "/records",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    def advance(self, step):
        stamp = self.base + step * SECOND
        self.now.return_value = stamp
        self.engine.advance(stamp)

    def bootstrap(self):
        for step in range(140):
            self.append(step)
        self.advance(139)
        deadline = time.monotonic() + 35
        while self.engine._state()["challenger"] is None and time.monotonic() < deadline:
            self.advance(139)
            if not self.engine.store.active_job(self.engine.key):
                self.advance(139)  # Consume a result committed after this tick began.
                break
            time.sleep(.03)
        self.assertEqual(self.engine._state()["stage"], "INITIAL_APPROVAL",
                         self.engine.store.events(self.engine.key)[-5:])
        self.assertIsNone(self.engine._state()["champion"])
        self.engine.approve(current_actor())
        self.advance(139)

    def test_real_worker_serving_mature_labels_monitor_and_restart(self):
        self.bootstrap()
        first = streams.predictions(self.engine.runtime, self.engine.key)
        self.assertEqual((first["origins"], len(first["predictions"])), (1, 2))
        champion = self.engine._state()["champion"]
        for step in range(140, 149):
            self.append(step, delay=3 if step == 140 else 0)
            self.advance(step)
            if step == 140:
                with self.engine.store.connection() as db:
                    count = db.execute("SELECT COUNT(*) FROM labels WHERE task_key=? AND target_ns=?",
                        (self.engine.key, self.base + 140 * SECOND)).fetchone()[0]
                self.assertEqual(count, 0)
        overview = streams.status(self.engine.runtime, self.engine.key)
        self.assertGreaterEqual(overview["counts"]["monitoring"], 1)
        self.assertTrue(overview["latest_monitoring_windows"])
        with patch("online_ts.engine.degradation", return_value=({"triggered": True}, True)):
            # A persisted triggering window must recover the pending action too.
            restored = self.engine._evaluate_mature_windows(champion, self.base + 148 * SECOND,
                                                            track_degradation=True)
        self.assertTrue(restored[1])
        with self.engine.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM labels WHERE task_key=? AND target_ns=?",
                (self.engine.key, self.base + 140 * SECOND)).fetchone()[0], 1)
        # Start a second actual training job; serving is available while its child runs.
        self.assertIsNotNone(self.engine.retrain())
        self.append(149)
        self.advance(149)
        self.assertEqual(self.engine._state()["champion"], champion)
        cursor = first["next_after_issue"]
        subsequent = streams.predictions(self.engine.runtime, self.engine.key, after_issue=cursor)
        self.assertEqual(subsequent["origins"], 10)
        total = streams.status(self.engine.runtime, self.engine.key)["counts"]["predictions"]
        # Simulate a crash after durable windows but before derived summary commit.
        self.engine.store.change(self.engine.key, lambda s: s.update(last_check=None,
            monitor={"stale_summary_from_interrupted_commit": True}))
        self.engine.close()
        self.engine = Engine(self.root / "runtime", self.engine.key)
        self.advance(149)
        self.assertEqual(streams.status(self.engine.runtime, self.engine.key)["counts"]["predictions"], total)
        self.assertNotIn("stale_summary_from_interrupted_commit", self.engine._state()["monitor"])
        self.engine.stop()
        self.assertTrue(self.engine._state()["stop_requested"])
        self.engine.resume()
        self.append(150)
        self.advance(150)
        self.assertEqual(streams.predictions(self.engine.runtime, self.engine.key,
            after_issue=subsequent["next_after_issue"])["origins"], 1)

    def test_old_delayed_and_revised_labels_survive_rolling_input_window(self):
        self.bootstrap()
        self.append(140, delay=200)
        self.advance(140)
        for step in range(141, 342):
            self.append(step)
        self.advance(341)
        with self.engine.store.connection() as db:
            old = db.execute("SELECT * FROM labels WHERE task_key=? AND target_ns=?",
                (self.engine.key, self.base + 140 * SECOND)).fetchone()
        self.assertIsNotNone(old)
        self.append(140, key="revision-140", value=99., received_step=342, delay=5)
        self.advance(342)
        prior = self.engine.store.prediction_rows(self.engine.key, self.engine._state()["champion"],
            self.base + 139 * SECOND, self.base + 140 * SECOND, self.base + 341 * SECOND)
        current = self.engine.store.prediction_rows(self.engine.key, self.engine._state()["champion"],
            self.base + 139 * SECOND, self.base + 140 * SECOND, self.base + 342 * SECOND)
        self.assertIsNotNone(prior[0]["actual"])
        self.assertIsNone(current[0]["actual"])
        self.advance(347)
        with self.engine.store.connection() as db:
            revised = db.execute("SELECT value FROM labels WHERE task_key=? AND target_ns=?",
                (self.engine.key, self.base + 140 * SECOND)).fetchone()
        self.assertEqual(revised[0], 99.)
        # No fresh inputs remain, yet withdrawal and later truth must still settle.
        self.append(140, key="correction-after-producer-stopped", value=101.,
                    received_step=600, delay=5)
        self.advance(600)
        self.advance(605)
        stopped = self.engine.store.prediction_rows(self.engine.key, self.engine._state()["champion"],
            self.base + 139 * SECOND, self.base + 140 * SECOND, self.base + 605 * SECOND)
        self.assertEqual(stopped[0]["actual"], 101.)
        self.assertEqual(self.engine._state()["status"], "DEGRADED")

    def test_status_does_not_create_runtime_and_runner_heartbeat(self):
        from online_ts.config import GateError
        missing = self.root / "absent"
        with self.assertRaisesRegex(GateError, "STREAM_RUNTIME_NOT_FOUND"):
            streams.status(missing, "task")
        self.assertFalse(missing.exists())
        target = self.root / "config.json"
        target.write_text(json.dumps(config().to_dict()), encoding="utf-8")
        result = streams.run(target, self.engine.runtime, max_ticks=1)
        self.assertEqual(result["runner"]["runner_state"], "paused")
        self.assertFalse(result["runner"]["heartbeat_fresh"])
        self.assertEqual(result["runner"]["ticks"], 1)


if __name__ == "__main__":
    unittest.main()
