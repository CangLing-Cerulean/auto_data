"""Tiny deterministic workflow example; no external data or model API."""
import json
from pathlib import Path
import statistics
import sys


def main():
    action, context_file = sys.argv[1:]
    context = json.loads(Path(context_file).read_text(encoding="utf-8"))
    output = Path(context["attempt"])
    if action == "measure":
        result = {"values": [1., 2., 3., 4.], "kind": "synthetic"}
        name = "measurements.json"
    elif action == "evaluate":
        measured = Path(context["dependencies"]["measure"]) / "measurements.json"
        values = json.loads(measured.read_text(encoding="utf-8"))["values"]
        result = {"mean": statistics.mean(values), "count": len(values),
                  "study_sha256": context["study_sha256"]}
        name = "metrics.json"
    else:
        raise ValueError(action)
    with (output / name).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)


if __name__ == "__main__":
    main()
