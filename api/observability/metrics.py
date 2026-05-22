from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    timestamp: str
    session_id: str
    latency_ms: float
    input_chars: int
    output_chars: int
    estimated_tokens: int
    estimated_cost_usd: float
    guardrail_blocks: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None
    inference_latency_ms: float | None = None
    ttft_ms: float | None = None
    tbt_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_sec: float | None = None


class RequestMetricsStore:
    def __init__(self, max_rows: int = 5000) -> None:
        self._rows: list[RequestMetric] = []
        self._max_rows = max_rows
        self._lock = threading.Lock()

    def record(self, metric: RequestMetric) -> None:
        with self._lock:
            self._rows.append(metric)
            if len(self._rows) > self._max_rows:
                self._rows = self._rows[-self._max_rows :]

    def snapshot(self) -> list[RequestMetric]:
        with self._lock:
            return list(self._rows)

    def cost_latency_table(
        self,
        *,
        cpu_hour_usd: float,
        tokens_per_char: float,
        fallback_latency_ms: float = 1500.0,
    ) -> dict:
        from api.observability.cost_model import build_cost_latency_table

        return build_cost_latency_table(
            self.snapshot(),
            cpu_hour_usd=cpu_hour_usd,
            tokens_per_char=tokens_per_char,
            fallback_latency_ms=fallback_latency_ms,
        )

    def inference_summary(self) -> dict[str, float | int]:
        from api.observability.inference_metrics import InferenceMetrics, summarize_inference

        rows = [
            InferenceMetrics(
                latency_ms=row.inference_latency_ms or row.latency_ms,
                ttft_ms=row.ttft_ms,
                tbt_ms=row.tbt_ms,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                stream_chunks=0,
                tokens_per_sec=row.tokens_per_sec,
            )
            for row in self.snapshot()
            if row.inference_latency_ms is not None or row.ttft_ms is not None
        ]
        return summarize_inference(rows)


METRICS_STORE = RequestMetricsStore()


def estimate_cost_usd(
    latency_ms: float,
    text_chars: int,
    cpu_hour_usd: float,
    tokens_per_char: float,
) -> tuple[int, float]:
    from api.observability.cost_model import estimate_request_cost_usd

    return estimate_request_cost_usd(
        latency_ms=latency_ms,
        total_chars=text_chars,
        cpu_hour_usd=cpu_hour_usd,
        tokens_per_char=tokens_per_char,
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
