from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .config import GateError, SourceConfig, TaskConfig, canonical


@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    size: int
    media_type: str = "application/json"

    def to_dict(self) -> dict:
        return {"sha256": self.sha256, "size": self.size, "media_type": self.media_type}


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, payload: bytes, media_type: str = "application/json") -> ArtifactRef:
        sha = hashlib.sha256(payload).hexdigest()
        path = self.root / sha[:2] / sha
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(name, path)
            finally:
                Path(name).unlink(missing_ok=True)
        ref = ArtifactRef(sha, len(payload), media_type)
        self.read(ref)
        return ref

    def put_json(self, value: dict) -> ArtifactRef:
        return self.put(canonical(value).encode("utf-8"))

    def read(self, ref: ArtifactRef | dict) -> bytes:
        if isinstance(ref, dict):
            ref = ArtifactRef(**ref)
        if len(ref.sha256) != 64 or any(c not in "0123456789abcdef" for c in ref.sha256):
            raise GateError("ARTIFACT_REFERENCE")
        path = (self.root / ref.sha256[:2] / ref.sha256).resolve()
        if not path.is_relative_to(self.root):
            raise GateError("ARTIFACT_PATH")
        try:
            payload = path.read_bytes()
        except OSError:
            raise GateError("ARTIFACT_MISSING") from None
        if len(payload) != ref.size or hashlib.sha256(payload).hexdigest() != ref.sha256:
            raise GateError("ARTIFACT_TAMPERED")
        return payload

    def read_json(self, ref: ArtifactRef | dict) -> dict:
        return json.loads(self.read(ref))


def time_ns(value) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise GateError("UTC_OFFSET_REQUIRED")
    return int(stamp.value)


def iso_time(ns: int) -> str:
    return pd.Timestamp(ns, unit="ns", tz="UTC").isoformat()


def normalize(frame: pd.DataFrame, config: TaskConfig, *, check_data: bool = True) -> pd.DataFrame:
    columns = [config.source.timestamp_column, *config.variables]
    if any(c not in frame.columns for c in columns):
        raise GateError("SOURCE_FIELDS_MISSING")
    try:
        stamps = pd.to_datetime(frame[columns[0]], utc=True, errors="raise").astype("datetime64[ns, UTC]")
        index = pd.Index(stamps.astype("int64"), name="time")
        values = frame[list(config.variables)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError):
        raise GateError("SOURCE_SCHEMA") from None
    if check_data and index.has_duplicates:
        raise GateError("DUPLICATE_TIMESTAMPS")
    if check_data and not index.is_monotonic_increasing:
        raise GateError("TIME_NOT_ORDERED")
    if check_data and np.isinf(values).any():
        raise GateError("NON_FINITE_VALUES")
    result = pd.DataFrame(values, index=index, columns=config.variables)
    available = config.source.available_at_column
    if available:
        if available not in frame:
            raise GateError("AVAILABILITY_COLUMN_MISSING")
        try:
            arrival = pd.to_datetime(frame[available], utc=True, errors="raise").astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        except (ValueError, TypeError):
            raise GateError("AVAILABILITY_SCHEMA") from None
        if np.any(arrival < index.to_numpy()):
            raise GateError("ARRIVAL_BEFORE_EVENT")
        result["__available_at"] = arrival
    else:
        result["__available_at"] = index.to_numpy()
    label_column = config.source.label_available_at_column
    if label_column:
        if label_column not in frame:
            raise GateError("LABEL_AVAILABILITY_COLUMN_MISSING")
        try:
            arrival = pd.to_datetime(frame[label_column], utc=True, errors="raise").astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        except (ValueError, TypeError):
            raise GateError("LABEL_AVAILABILITY_SCHEMA") from None
        if np.any(arrival < result["__available_at"].to_numpy()):
            raise GateError("LABEL_COMPLETE_BEFORE_OBSERVATION")
        result["__label_available_at"] = arrival
    else:
        result["__label_available_at"] = result["__available_at"]
    revision_column = config.source.revision_at_column
    if revision_column:
        if revision_column not in frame:
            raise GateError("REVISION_COLUMN_MISSING")
        try:
            revised = pd.to_datetime(frame[revision_column], utc=True, errors="raise").astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        except (ValueError, TypeError):
            raise GateError("REVISION_SCHEMA") from None
        if np.any(revised < result["__available_at"].to_numpy()):
            raise GateError("REVISION_BEFORE_ARRIVAL")
        result["__revision_at"] = revised
    else:
        result["__revision_at"] = result["__available_at"]
    if config.source.regime_column:
        if config.source.regime_column not in frame:
            raise GateError("REGIME_COLUMN_MISSING")
        result["__regime"] = [str(v) if pd.notna(v) and str(v).strip() else None
                              for v in frame[config.source.regime_column]]
    return result


class DataSourceAdapter(Protocol):
    def read_until(self, now: int, start: int | None = None) -> pd.DataFrame: ...
    def discover(self, now: int) -> dict: ...


class ReplaySource:
    """Only this adapter may hold unreleased observations; consumers use read_until."""

    def __init__(self, config: TaskConfig, artifacts: ArtifactStore, ref: dict | None = None):
        self.config = config
        self.artifacts = artifacts
        if ref is None:
            path = Path(config.source.path).resolve(strict=True)
            media = "text/csv" if path.suffix.lower() == ".csv" else "application/parquet"
            if path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
                raise GateError("SOURCE_FORMAT")
            self.ref = artifacts.put(path.read_bytes(), media).to_dict()
        else:
            self.ref = ref
        payload = artifacts.read(self.ref)
        raw = pd.read_csv(io.BytesIO(payload)) if self.ref["media_type"] == "text/csv" else pd.read_parquet(io.BytesIO(payload))
        self.__frame = normalize(raw, config, check_data=False)
        if self.__frame.empty:
            raise GateError("SOURCE_EMPTY")
        pairs = list(zip(self.__frame.index.to_numpy(), self.__frame["__revision_at"].to_numpy()))
        if len(pairs) != len(set(pairs)):
            raise GateError("DUPLICATE_REVISION")

    @property
    def bounds(self) -> tuple[int, int]:
        # Used only by the clock/bootstrap coordinator, never by a model/planner.
        return int(self.__frame.index.min()), int(self.__frame.index.max())

    def read_until(self, now: int, start: int | None = None) -> pd.DataFrame:
        self.artifacts.read(self.ref)
        visible = ((self.__frame.index <= now) & (self.__frame["__available_at"] <= now)
                   & (self.__frame["__revision_at"] <= now))
        if start is not None:
            visible &= self.__frame.index >= start
        selected = self.__frame.loc[visible].sort_values("__revision_at", kind="stable")
        selected = selected.loc[~selected.index.duplicated(keep="last")].sort_index()
        result = selected.loc[:, list(self.config.variables)].copy()
        if result.index.has_duplicates or not result.index.is_monotonic_increasing:
            raise GateError("TIME_ORDER_OR_DUPLICATE")
        if np.isinf(result.to_numpy()).any():
            raise GateError("NON_FINITE_VALUES")
        result.attrs["as_of"] = now
        result.attrs["max_available_at"] = int(selected["__available_at"].max()) if len(result) else now
        result.attrs["max_revision_at"] = int(selected["__revision_at"].max()) if len(result) else now
        result.attrs["revision_visibility"] = "explicit" if self.config.source.revision_at_column else "not_provided"
        result.attrs["availability_by_time"] = dict(zip(selected.index.tolist(), np.maximum(
            selected["__available_at"].to_numpy(), selected["__revision_at"].to_numpy()).tolist()))
        result.attrs["label_availability_by_time"] = dict(zip(selected.index.tolist(), np.maximum(
            selected["__label_available_at"].to_numpy(), selected["__revision_at"].to_numpy()).tolist()))
        if self.config.source.regime_column:
            result.attrs["regime_by_time"] = dict(zip(selected.index.tolist(), selected["__regime"].tolist()))
        return result

    def labels_until(self, now: int, start: int | None = None) -> pd.DataFrame:
        visible = ((self.__frame.index <= now) & (self.__frame["__available_at"] <= now)
                   & (self.__frame["__revision_at"] <= now) & (self.__frame["__label_available_at"] <= now))
        if start is not None:
            visible &= self.__frame.index >= start
        selected = self.__frame.loc[visible].sort_values("__revision_at", kind="stable")
        selected = selected.loc[~selected.index.duplicated(keep="last")].sort_index()
        result = selected.loc[:, list(self.config.variables)].copy()
        result.attrs["as_of"] = now
        result.attrs["max_available_at"] = int(selected["__label_available_at"].max()) if len(result) else now
        result.attrs["max_revision_at"] = int(selected["__revision_at"].max()) if len(result) else now
        result.attrs["revision_visibility"] = "explicit" if self.config.source.revision_at_column else "not_provided"
        result.attrs["availability_by_time"] = dict(zip(selected.index.tolist(), np.maximum(
            selected["__available_at"].to_numpy(), selected["__revision_at"].to_numpy()).tolist()))
        result.attrs["label_availability_by_time"] = dict(zip(selected.index.tolist(), np.maximum(
            selected["__label_available_at"].to_numpy(), selected["__revision_at"].to_numpy()).tolist()))
        if self.config.source.regime_column:
            result.attrs["regime_by_time"] = dict(zip(selected.index.tolist(), selected["__regime"].tolist()))
        return result

    def discover(self, now: int) -> dict:
        return profile(self.read_until(now), self.config)


def aligned(frame: pd.DataFrame, config: TaskConfig, *, end: int | None = None) -> pd.DataFrame:
    if frame.empty:
        raise GateError("DATA_INSUFFICIENT")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise GateError("TIME_ORDER_OR_DUPLICATE")
    times = frame.index.to_numpy(dtype=np.int64)
    if np.any((times - times[0]) % config.step_ns):
        raise GateError("FREQUENCY_MISMATCH")
    last = int(times[-1]) if end is None else end
    if last < times[0] or (last - times[0]) % config.step_ns:
        raise GateError("WATERMARK_ALIGNMENT")
    index = pd.Index(np.arange(times[0], last + config.step_ns, config.step_ns, dtype=np.int64), name="time")
    # Carry metadata once, after the value operations; pandas otherwise deep
    # copies the complete availability maps at each intermediate operation.
    result = pd.DataFrame(frame, copy=False).reindex(index)[list(config.variables)]
    result.attrs = dict(frame.attrs)
    if np.isinf(result.to_numpy()).any():
        raise GateError("NON_FINITE_VALUES")
    return result


def causal_fill(frame: pd.DataFrame, config: TaskConfig) -> pd.DataFrame:
    return frame.ffill(limit=config.max_fill_steps) if config.max_fill_steps else frame.copy()


def profile(frame: pd.DataFrame, config: TaskConfig) -> dict:
    grid = aligned(frame, config)
    values = grid.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    longest = []
    for column in missing.T:
        run = best = 0
        for absent in column:
            run = run + 1 if absent else 0
            best = max(run, best)
        longest.append(best)
    return {
        "rows": len(grid), "start": int(grid.index[0]), "end": int(grid.index[-1]),
        "variables": list(config.variables), "frequency_seconds": config.frequency_seconds,
        "observed_interval_seconds": float(np.median(np.diff(frame.index))) / 1e9 if len(frame) > 1 else None,
        "missing_fraction": float(missing.mean()), "missing_by_variable": missing.mean(axis=0).tolist(),
        "longest_missing_steps": longest,
        "constant_variables": [v for i, v in enumerate(config.variables) if np.unique(values[~missing[:, i], i]).size <= 1],
    }


def snapshot(frame: pd.DataFrame, cutoff: int, config: TaskConfig, artifacts: ArtifactStore, *, as_of: int | None = None) -> ArtifactRef:
    if frame.empty or int(frame.index.max()) > cutoff:
        raise GateError("SNAPSHOT_FUTURE_LEAKAGE")
    as_of = cutoff if as_of is None else as_of
    latest_arrival = int(frame["__available_at"].max()) if "__available_at" in frame else int(frame.attrs.get("max_available_at", cutoff))
    latest_revision = int(frame["__revision_at"].max()) if "__revision_at" in frame else int(frame.attrs.get("max_revision_at", latest_arrival))
    if cutoff > as_of or latest_arrival > as_of or latest_revision > as_of:
        raise GateError("SNAPSHOT_ARRIVAL_LEAKAGE")
    grid = aligned(frame, config)
    from .qualification import input_qualification
    qualification = input_qualification(frame, config)
    if not qualification["input_eligible"]:
        raise GateError(qualification["issues"][0])
    report = qualification["profile"]
    return artifacts.put_json({
        "schema": 3, "config_version": config.version, "cutoff": cutoff, "as_of": as_of, "max_available_at": latest_arrival,
        "max_revision_at": latest_revision, "revision_visibility": frame.attrs.get("revision_visibility", "not_provided"),
        "variables": list(config.variables), "times": [int(t) for t in grid.index],
        "values": [[None if not np.isfinite(v) else float(v) for v in row] for row in grid.to_numpy()],
        "profile": report,
        "qualification": qualification,
        "regime_by_time": [frame.attrs.get("regime_by_time", {}).get(int(t)) for t in grid.index],
        "availability_by_time": [int(frame.attrs.get("availability_by_time", {}).get(int(t), t)) for t in grid.index],
        "label_availability_by_time": [int(frame.attrs.get("label_availability_by_time", {}).get(int(t), t)) for t in grid.index],
        "historical_availability": frame.attrs.get("historical_availability", "event_or_explicit"),
    })


def load_snapshot(ref: dict, config: TaskConfig, artifacts: ArtifactStore) -> tuple[pd.DataFrame, dict]:
    value = artifacts.read_json(ref)
    if value.get("schema") != 3 or value["config_version"] != config.version or value["variables"] != list(config.variables):
        raise GateError("SNAPSHOT_CONTRACT")
    if value["cutoff"] > value["as_of"] or value["max_available_at"] > value["as_of"] or value.get("max_revision_at", value["max_available_at"]) > value["as_of"]:
        raise GateError("SNAPSHOT_ARRIVAL_LEAKAGE")
    frame = pd.DataFrame(value["values"], index=pd.Index(value["times"], name="time"), columns=config.variables, dtype=float)
    if int(frame.index.max()) > value["cutoff"]:
        raise GateError("SNAPSHOT_FUTURE_LEAKAGE")
    frame = aligned(frame, config)
    if value.get("historical_availability") == "unverified":
        raise GateError("HISTORICAL_AVAILABILITY_UNVERIFIED")
    for name in ("availability_by_time", "label_availability_by_time"):
        timestamps = value.get(name, value["times"])
        if len(timestamps) != len(frame) or any(type(t) is not int for t in timestamps):
            raise GateError("SNAPSHOT_AVAILABILITY_SCHEMA")
        frame.attrs[name] = dict(zip((int(t) for t in frame.index), timestamps))
    frame.attrs["training_asof"] = value["as_of"]
    frame.attrs["evaluation_asof"] = value["as_of"]
    frame.attrs["regime_by_time"] = dict(zip(value["times"], value["regime_by_time"]))
    return frame, value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise GateError("NETWORK_REDIRECT_REFUSED")


def request_bytes(request: urllib.request.Request, timeout: float) -> bytes:
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(request, timeout=timeout) as response:
        body = response.read(64 * 1024 * 1024 + 1)
    if len(body) > 64 * 1024 * 1024:
        raise GateError("SOURCE_RESPONSE_TOO_LARGE")
    return body


class InfluxSource:
    """Fixed read-only Flux queries; no user-supplied Flux or write endpoint."""

    def __init__(self, config: TaskConfig, transport: Callable = request_bytes, sleeper: Callable = time.sleep):
        self.config = config
        self.transport = transport
        self.sleeper = sleeper

    def _query(self, query: str) -> list[dict]:
        source: SourceConfig = self.config.source
        url = os.environ.get(source.url_env, "")
        token = os.environ.get(source.token_env, "")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise GateError("INFLUX_URL_CONFIGURATION")
        if not token:
            raise GateError("INFLUX_CREDENTIALS_MISSING")
        endpoint = url.rstrip("/") + "/api/v2/query?" + urllib.parse.urlencode({"org": source.org})
        request = urllib.request.Request(endpoint, data=query.encode("utf-8"), method="POST", headers={
            "Authorization": "Token " + token, "Accept": "application/csv", "Content-Type": "application/vnd.flux",
        })
        for attempt in range(source.retries):
            try:
                payload = self.transport(request, source.timeout_seconds)
                break
            except urllib.error.HTTPError as error:
                error.close()
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    raise GateError("INFLUX_QUERY_REJECTED") from None
                if attempt == source.retries - 1:
                    raise GateError("INFLUX_UNAVAILABLE") from None
                self.sleeper(min(2 ** attempt, 4))
            except (OSError, TimeoutError):
                if attempt == source.retries - 1:
                    raise GateError("INFLUX_UNAVAILABLE") from None
                self.sleeper(min(2 ** attempt, 4))
        rows: list[dict] = []
        header = None
        try:
            for row in csv.reader(io.StringIO(payload.decode("utf-8"))):
                if not row or row[0].startswith("#"):
                    continue
                if "_time" in row and "_field" in row:
                    header = row
                elif "error" in row and "reference" in row:
                    raise GateError("INFLUX_QUERY_REJECTED")
                elif header is not None:
                    if len(row) != len(header):
                        raise GateError("INFLUX_RESPONSE_SCHEMA")
                    rows.append(dict(zip(header, row)))
        except (UnicodeDecodeError, csv.Error):
            raise GateError("INFLUX_RESPONSE_SCHEMA") from None
        return rows

    def _base(self, start: int, now: int, *, selected: bool = True) -> str:
        src = self.config.source
        quote = lambda s: json.dumps(str(s), ensure_ascii=False)
        flux = f"from(bucket: {quote(src.bucket)}) |> range(start: time(v: {quote(iso_time(start))}), stop: time(v: {quote(iso_time(now + 1))}))"
        flux += f" |> filter(fn: (r) => r._measurement == {quote(src.measurement)})"
        for key, value in sorted(src.tags.items()):
            flux += f" |> filter(fn: (r) => r[{quote(key)}] == {quote(value)})"
        if selected:
            expression = " or ".join(f"r._field == {quote(v)}" for v in self.config.variables)
            flux += f" |> filter(fn: (r) => {expression})"
        return flux

    def read_until(self, now: int, start: int | None = None) -> pd.DataFrame:
        if start is None:
            start = now - self.config.search.required_history * self.config.step_ns
        rows = self._query(self._base(start, now) + ' |> group() |> sort(columns: ["_time"])')
        records: dict[int, dict] = {}
        seen: set[tuple[int, str]] = set()
        for row in rows:
            try:
                stamp = time_ns(row["_time"])
                variable = row["_field"]
                value = float(row["_value"])
            except (KeyError, ValueError, TypeError):
                raise GateError("INFLUX_RESPONSE_SCHEMA") from None
            if stamp > now or stamp < start or variable not in self.config.variables:
                raise GateError("INFLUX_QUERY_BOUNDARY")
            if (stamp, variable) in seen:
                raise GateError("INFLUX_AMBIGUOUS_SERIES")
            seen.add((stamp, variable))
            records.setdefault(stamp, {})[variable] = value
        if not records:
            return pd.DataFrame(columns=self.config.variables, index=pd.Index([], name="time", dtype="int64"), dtype=float)
        frame = pd.DataFrame.from_dict(records, orient="index").reindex(columns=self.config.variables).sort_index().rename_axis("time")
        frame.attrs.update(as_of=now, max_available_at=now, historical_availability="unverified")
        return frame

    def discover(self, now: int) -> dict:
        base = self._base(0, now, selected=False)
        first = self._query(base + ' |> group(columns: ["_field"]) |> first()')
        last = self._query(base + ' |> group(columns: ["_field"]) |> last()')
        if not first or not last:
            raise GateError("SOURCE_EMPTY")
        fields = sorted({r["_field"] for r in first + last})
        if set(self.config.variables) - set(fields):
            raise GateError("SOURCE_FIELDS_MISSING")
        report = profile(self.read_until(now), self.config)
        report.update({"source_min_time": min(time_ns(r["_time"]) for r in first), "source_max_time": max(time_ns(r["_time"]) for r in last), "available_fields": fields})
        return report
