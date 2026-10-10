"""Bounded streaming with request-level timing, cancellation and partial progress."""
from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import GateError, canonical, digest
from .data import NoRedirect


MAX_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass
class LLMCallTrace:
    call_id: str
    role: str
    callback: object = field(default=None, repr=False)
    started: float = field(default_factory=time.monotonic, repr=False)
    data: dict = field(default_factory=dict)
    last_emit: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self.data.update(schema_version=1, call_id=self.call_id, role=self.role,
                         started_ns=time.time_ns(), state="preparing", received_bytes=0,
                         received_content_chars=0, reasoning_chars=0, usage=None,
                         ttft_ms=None, tpot_ms=None, server_queue_ms=None,
                         timing_scope="client_observed_sse_chunks_not_individual_tokens")

    def mark(self, name, value=None):
        self.data[name] = round((time.monotonic() - self.started) * 1000, 3) if value is None else value

    def emit(self, *, force=False):
        now = time.monotonic()
        if self.callback is not None and (force or now - self.last_emit >= 1):
            self.last_emit = now
            self.data["elapsed_ms"] = round((now - self.started) * 1000, 3)
            self.data["remaining_seconds"] = max(0.0, self.data.get("timeout_seconds", 0) - (now - self.started))
            self.callback(dict(self.data))

    def prepare(self, request, timeout, prompt_reserve, output_tokens, omitted=()):
        body = json.loads(request.data)
        self.data.update(state="prepared", timeout_seconds=timeout,
                         request_bytes=len(request.data), request_fingerprint=digest(body),
                         model_requested=body["model"], messages=body["messages"],
                         prompt_reserved_tokens=prompt_reserve, output_limit=output_tokens,
                         reservation_method="utf8_bytes_plus_256_framing_not_actual_token_count",
                         omitted_background=list(omitted),
                         requested_reasoning_effort=body.get("reasoning_effort"),
                         thinking_setting_verified=False)
        self.mark("prepared_ms")

    def finish(self, error=None):
        self.data.update(state="failed" if error else "completed", error_code=error)
        self.mark("total_ms")
        self.emit(force=True)


def read_chat_stream(response, deadline: float, *, trace=None, cancelled=None) -> bytes:
    content = []
    usage = None
    model = None
    finish = None
    total = 0
    done = False
    while True:
        if cancelled is not None and cancelled():
            raise GateError("PLANNER_CANCELLED")
        if time.monotonic() > deadline:
            raise GateError("PLANNER_STREAM_DEADLINE")
        raw = response.readline(MAX_RESPONSE_BYTES - total + 1)
        if not raw:
            break
        total += len(raw)
        if total > MAX_RESPONSE_BYTES:
            raise GateError("PLANNER_RESPONSE_TOO_LARGE")
        if time.monotonic() > deadline:
            raise GateError("PLANNER_STREAM_DEADLINE")
        if trace is not None:
            trace.data.update(received_bytes=total, last_response_ns=time.time_ns(), state="receiving")
            trace.emit()
        line = raw.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise GateError("PLANNER_STREAM_SCHEMA")
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        chunk = json.loads(payload)
        if not isinstance(chunk, dict) or chunk.get("error"):
            raise GateError("PLANNER_STREAM_SCHEMA")
        model = chunk.get("model", model)
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
            if trace is not None:
                trace.data["usage"] = usage
        if trace is not None:
            trace.data.update(model_returned=model, system_fingerprint=chunk.get("system_fingerprint", trace.data.get("system_fingerprint")))
        for choice in chunk.get("choices", []):
            if choice.get("index") != 0:
                raise GateError("PLANNER_STREAM_SCHEMA")
            delta = choice.get("delta", {})
            if delta.get("tool_calls") or delta.get("function_call"):
                raise GateError("PLANNER_STREAM_SCHEMA")
            text, reasoning = delta.get("content"), delta.get("reasoning_content")
            if text is not None and not isinstance(text, str):
                raise GateError("PLANNER_STREAM_SCHEMA")
            if reasoning is not None and not isinstance(reasoning, str):
                raise GateError("PLANNER_STREAM_SCHEMA")
            if trace is not None and (text or reasoning):
                if "first_nonempty_delta_ms" not in trace.data:
                    trace.mark("first_nonempty_delta_ms")
                if reasoning:
                    trace.data["reasoning_chars"] += len(reasoning)
                if text:
                    if "first_answer_content_ms" not in trace.data:
                        trace.mark("first_answer_content_ms")
                    trace.mark("last_answer_content_ms")
                    trace.data["received_content_chars"] += len(text)
            if text is not None:
                content.append(text)
                if trace is not None:
                    trace.data["model_response"] = "".join(content)
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
        if trace is not None:
            trace.emit()
    if trace is not None:
        trace.data.update(finish_reason=finish, stream_done=done, model_response="".join(content))
        trace.mark("stream_end_ms")
    # A completed wire stream can contain an unfinished model answer. Preserve
    # its finish reason and usage so the caller rejects it without network retry
    # and charges the actual inference, including reasoning tokens.
    if not done or finish not in {"stop", "length"} or usage is None:
        raise GateError("PLANNER_STREAM_INCOMPLETE")
    return canonical({"model": model, "usage": usage,
                      "choices": [{"message": {"content": "".join(content)}, "finish_reason": finish}]}).encode()


def _request_chat_bytes(request: urllib.request.Request, timeout: float, *, trace=None) -> bytes:
    body = json.loads(request.data)
    body.update(stream=True, stream_options={"include_usage": True})
    serialize_start = time.monotonic()
    payload = canonical(body).encode()
    streaming = urllib.request.Request(request.full_url, data=payload,
                                      headers=dict(request.header_items()), method="POST")
    if trace is not None:
        trace.data.update(serialization_ms=round((time.monotonic() - serialize_start) * 1000, 3),
                          request_bytes=len(payload), request_fingerprint=digest(body), state="connecting")
        trace.mark("request_start_ms")
        trace.emit(force=True)
    deadline = time.monotonic() + timeout
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect).open(streaming, timeout=timeout) as response:
            if trace is not None:
                trace.data["http_status"] = response.status
                trace.mark("response_headers_ms")
                trace.data["state"] = "waiting_content"
                trace.emit(force=True)
            if response.headers.get_content_type() != "text/event-stream":
                raise GateError("PLANNER_STREAM_CONTENT_TYPE")
            return read_chat_stream(response, deadline, trace=trace)
    except urllib.error.HTTPError as error:
        if trace is not None:
            trace.data["http_status"] = error.code
        raise GateError("PLANNER_HTTP_AUTH" if error.code in {401, 403} else
                        "PLANNER_GATEWAY_TIMEOUT" if error.code == 504 else "PLANNER_HTTP_ERROR") from None
    except GateError:
        raise
    except (OSError, ValueError, UnicodeError):
        raise GateError("PLANNER_STREAM_DEADLINE" if time.monotonic() >= deadline else "PLANNER_TRANSPORT_FAILED") from None


def _network_process(channel, request_fields, timeout, trace_fields):
    """Own the socket in a killable process, including DNS and response headers."""
    parent = multiprocessing.parent_process()
    def stop_with_parent():
        while parent is not None and parent.is_alive():
            time.sleep(.25)
        os._exit(0)
    threading.Thread(target=stop_with_parent, daemon=True).start()
    trace = LLMCallTrace(trace_fields["call_id"], trace_fields["role"],
                         callback=lambda data: channel.send(("trace", data)))
    trace.started = trace_fields["started"]
    trace.data = trace_fields["data"]
    try:
        request = urllib.request.Request(**request_fields)
        result = _request_chat_bytes(request, timeout, trace=trace)
        trace.emit(force=True)
        channel.send(("result", result))
    except Exception as error:
        trace.emit(force=True)
        channel.send(("error", str(error) if isinstance(error, GateError) else "PLANNER_TRANSPORT_FAILED"))
    finally:
        channel.close()


def request_chat_bytes(request: urllib.request.Request, timeout: float, *, trace=None, cancelled=None) -> bytes:
    # Socket shutdown does not reliably interrupt blocked Windows reads. A child
    # owns only network I/O; the parent owns accounting, validation and all writes.
    # Terminate and join it before releasing the caller's single-request lease.
    trace = trace or LLMCallTrace("transport", "planning")
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_network_process, args=(writer,
        {"url": request.full_url, "data": request.data, "headers": dict(request.header_items()), "method": "POST"},
        timeout, {"call_id": trace.call_id, "role": trace.role, "started": trace.started, "data": dict(trace.data)}))
    deadline = time.monotonic() + timeout
    process.start()
    writer.close()
    try:
        while True:
            if cancelled is not None and cancelled():
                raise GateError("PLANNER_CANCELLED")
            if time.monotonic() >= deadline:
                raise GateError("PLANNER_STREAM_DEADLINE")
            if reader.poll(min(.25, max(0, deadline - time.monotonic()))):
                try:
                    kind, value = reader.recv()
                except EOFError:
                    raise GateError("PLANNER_TRANSPORT_FAILED") from None
                if kind == "trace":
                    trace.data.update(value)
                    trace.emit()
                elif kind == "error":
                    raise GateError(value)
                elif kind == "result":
                    return value
            trace.emit()
            if not process.is_alive() and not reader.poll():
                raise GateError("PLANNER_TRANSPORT_FAILED")
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        reader.close()
        process.close()
