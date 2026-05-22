from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


def _format_trace_id(value: int) -> str:
    return format(value, "032x")


def _format_span_id(value: int) -> str:
    return format(value, "016x")


def _serialize_attr(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return json.dumps([_serialize_attr(item) for item in value])
    return str(value)


def _iso_to_epoch_ms(iso: str) -> float | None:
    if not iso:
        return None
    try:
        normalized = iso.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp() * 1000
    except ValueError:
        return None


def _trace_wall_duration_ms(spans: list[StoredSpan]) -> float:
    """End-to-end trace time: earliest start to latest end (avoids double-counting nested spans)."""
    starts = [value for value in (_iso_to_epoch_ms(span.start_time) for span in spans) if value is not None]
    ends = [value for value in (_iso_to_epoch_ms(span.end_time) for span in spans) if value is not None]
    if starts and ends:
        return max(max(ends) - min(starts), 0.0)

    roots = [span for span in spans if not span.parent_span_id]
    fallback = roots or spans
    return max((span.duration_ms for span in fallback), default=0.0)


def _pick_root_span(spans: list[StoredSpan]) -> StoredSpan:
    roots = [span for span in spans if not span.parent_span_id]
    if roots:
        return max(roots, key=lambda span: span.duration_ms)
    return min(spans, key=lambda span: span.start_time or "")


def _parse_guardrail_blocks(chat_attrs: dict[str, str | int | float | bool]) -> list[str]:
    raw = chat_attrs.get("guardrail.blocks")
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _trace_status(trace_spans: list[StoredSpan], chat_attrs: dict[str, str | int | float | bool]) -> str:
    if any(span.status == "error" for span in trace_spans):
        return "error"
    if _parse_guardrail_blocks(chat_attrs) or any(span.status == "blocked" for span in trace_spans):
        return "blocked"
    return "ok"


def _group_spans_by_trace(spans: list[StoredSpan]) -> dict[str, list[StoredSpan]]:
    traces: dict[str, list[StoredSpan]] = {}
    for span in spans:
        traces.setdefault(span.trace_id, []).append(span)
    return traces


def _ns_to_iso(ns: int) -> str:
    if ns <= 0:
        return ""
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC).isoformat()


@dataclass
class StoredSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    service_name: str
    kind: str
    start_time: str
    end_time: str
    duration_ms: float
    status: str
    attributes: dict[str, str | int | float | bool] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "service_name": self.service_name,
            "kind": self.kind,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class TraceStore:
    def __init__(self, max_spans: int = 2000) -> None:
        self._spans: list[StoredSpan] = []
        self._max_spans = max_spans
        self._lock = threading.Lock()
        self._listeners: list[queue.Queue[bool]] = []
        self._listeners_lock = threading.Lock()

    def subscribe(self) -> queue.Queue[bool]:
        listener: queue.Queue[bool] = queue.Queue(maxsize=1)
        with self._listeners_lock:
            self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: queue.Queue[bool]) -> None:
        with self._listeners_lock:
            self._listeners = [item for item in self._listeners if item is not listener]

    def notify_update(self) -> None:
        with self._listeners_lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                while not listener.empty():
                    listener.get_nowait()
                listener.put_nowait(True)
            except queue.Full:
                pass

    def build_payload(self, *, limit: int = 40) -> dict[str, Any]:
        from api.main import get_api_config
        from api.observability.metrics import METRICS_STORE

        config = get_api_config()
        stats = self.stats()
        stats["inference"] = METRICS_STORE.inference_summary()
        fallback_latency = float(stats["inference"].get("latency_p50_ms") or 1500.0)
        stats["cost_latency"] = METRICS_STORE.cost_latency_table(
            cpu_hour_usd=config.cost.cpu_hour_usd,
            tokens_per_char=config.cost.estimated_tokens_per_char,
            fallback_latency_ms=fallback_latency,
        )
        return {
            "stats": stats,
            "traces": self.grouped_traces(limit=limit),
        }

    def record(self, span: ReadableSpan, *, notify: bool = True) -> None:
        ctx = span.get_span_context()
        if not ctx or not ctx.is_valid:
            return

        trace_id = _format_trace_id(ctx.trace_id)
        span_id = _format_span_id(ctx.span_id)
        parent = span.parent
        parent_span_id = (
            _format_span_id(parent.span_id) if parent and parent.is_valid else None
        )

        start_ns = span.start_time or 0
        end_ns = span.end_time or 0
        duration_ms = max((end_ns - start_ns) / 1_000_000, 0.0)

        attributes = {
            str(key): _serialize_attr(value)
            for key, value in (span.attributes or {}).items()
        }

        status = "ok"
        if attributes.get("guardrail.blocks"):
            status = "blocked"
        elif span.status.status_code.name == "ERROR":
            status = "error"
        elif span.status.status_code.name == "UNSET":
            status = "unset"

        resource = span.resource.attributes if span.resource else {}
        service_name = str(resource.get("service.name", "unknown"))

        events = [
            {
                "name": event.name,
                "time": _ns_to_iso(event.timestamp),
                "attributes": {
                    str(key): _serialize_attr(value)
                    for key, value in (event.attributes or {}).items()
                },
            }
            for event in span.events
        ]

        stored = StoredSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=span.name,
            service_name=service_name,
            kind=span.kind.name if span.kind else "INTERNAL",
            start_time=_ns_to_iso(start_ns),
            end_time=_ns_to_iso(end_ns),
            duration_ms=round(duration_ms, 2),
            status=status,
            attributes=attributes,
            events=events,
        )

        with self._lock:
            self._spans.append(stored)
            if len(self._spans) > self._max_spans:
                self._spans = self._spans[-self._max_spans :]

        if notify:
            self.notify_update()

    def snapshot(self, *, limit: int | None = None) -> list[StoredSpan]:
        with self._lock:
            rows = list(self._spans)
        if limit is not None:
            rows = rows[-limit:]
        return rows

    def by_trace_id(self, trace_id: str) -> list[StoredSpan]:
        normalized = trace_id.lower().removeprefix("0x")
        with self._lock:
            return [
                span
                for span in self._spans
                if span.trace_id.endswith(normalized) or span.trace_id == normalized
            ]

    def grouped_traces(self, *, limit: int = 50) -> list[dict[str, Any]]:
        spans = self.snapshot(limit=limit * 20)
        traces = _group_spans_by_trace(spans)

        groups: list[dict[str, Any]] = []
        for trace_id, trace_spans in traces.items():
            trace_spans.sort(key=lambda row: row.start_time)
            root = _pick_root_span(trace_spans)
            wall_ms = _trace_wall_duration_ms(trace_spans)
            chat_span = next((span for span in trace_spans if span.name == "chat.request"), None)
            chat_attrs = chat_span.attributes if chat_span else {}
            guardrail_blocks = _parse_guardrail_blocks(chat_attrs)
            groups.append(
                {
                    "trace_id": trace_id,
                    "root_span": root.name,
                    "service_name": root.service_name,
                    "start_time": root.start_time,
                    "span_count": len(trace_spans),
                    "total_duration_ms": round(wall_ms, 2),
                    "input_tokens": int(chat_attrs.get("inference.input_tokens", 0) or 0),
                    "output_tokens": int(chat_attrs.get("inference.output_tokens", 0) or 0),
                    "ttft_ms": chat_attrs.get("inference.ttft_ms"),
                    "tbt_ms": chat_attrs.get("inference.tbt_ms"),
                    "tokens_per_sec": chat_attrs.get("inference.tokens_per_sec"),
                    "guardrail_blocks": guardrail_blocks,
                    "status": _trace_status(trace_spans, chat_attrs),
                    "spans": [span.to_dict() for span in trace_spans],
                }
            )

        groups.sort(key=lambda row: row["start_time"], reverse=True)
        return groups[:limit]

    def stats(self) -> dict[str, Any]:
        spans = self.snapshot()
        if not spans:
            return {
                "total_spans": 0,
                "total_traces": 0,
                "avg_duration_ms": 0.0,
                "error_spans": 0,
            }

        traces = _group_spans_by_trace(spans)
        trace_durations = [_trace_wall_duration_ms(trace_spans) for trace_spans in traces.values()]
        avg_duration = sum(trace_durations) / len(trace_durations)
        return {
            "total_spans": len(spans),
            "total_traces": len(traces),
            "avg_duration_ms": round(avg_duration, 2),
            "error_spans": sum(1 for span in spans if span.status in {"error", "blocked"}),
        }


class InMemorySpanExporter(SpanExporter):
    def __init__(self, store: TraceStore) -> None:
        self._store = store

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        recorded = 0
        for span in spans:
            try:
                self._store.record(span, notify=False)
                recorded += 1
            except Exception as exc:
                logger.warning("Failed to record span %s: %s", span.name, exc)
        if recorded:
            self._store.notify_update()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


TRACE_STORE = TraceStore()
