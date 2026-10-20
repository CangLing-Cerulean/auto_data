"""Example external adapter: send one causal observation per wall-clock second."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from online_ts.data import request_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18761")
    parser.add_argument("--seconds", type=int, default=240)
    args = parser.parse_args()
    endpoint = urllib.parse.urlsplit(args.url)
    if (endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or endpoint.username
            or endpoint.password or endpoint.path not in ("", "/") or endpoint.query or endpoint.fragment):
        raise ValueError("A loopback journal URL is required")
    if args.seconds < 1:
        raise ValueError("seconds must be positive")
    producer = uuid.uuid4().hex
    previous = None
    sent = 0
    while sent < args.seconds:
        second = time.time_ns() // 1_000_000_000
        if second == previous:
            time.sleep(.05)
            continue
        record = {"idempotency_key": f"{producer}:{second}",
                  "event_ns": second * 1_000_000_000,
                  "values": {"signal": 10 + .03 * sent + .1 * math.sin(sent / 10)},
                  "maturity_delay_seconds": 0}
        body = json.dumps(record, allow_nan=False).encode()
        request = urllib.request.Request(args.url.rstrip("/") + "/records", data=body,
            headers={"Content-Type": "application/json"})
        # A retry must use this exact key and payload; never invent a new key.
        result = json.loads(request_bytes(request, 5))
        print(json.dumps({"sent": sent + 1, "receipt": result}), flush=True)
        previous = second
        sent += 1


if __name__ == "__main__":
    main()
