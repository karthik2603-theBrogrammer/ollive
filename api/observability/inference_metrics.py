from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceMetrics:
    latency_ms: float
    ttft_ms: float | None
    tbt_ms: float | None
    input_tokens: int
    output_tokens: int
    stream_chunks: int
    tokens_per_sec: float | None = None


def compute_tokens_per_sec(
    output_tokens: int,
    latency_ms: float,
    ttft_ms: float | None,
) -> float | None:
    """Output decode throughput: output tokens divided by generation window after TTFT."""
    if output_tokens <= 0 or latency_ms <= 0:
        return None

    decode_ms = latency_ms
    if ttft_ms is not None and 0 <= ttft_ms < latency_ms:
        decode_ms = latency_ms - ttft_ms
    if decode_ms <= 0:
        decode_ms = latency_ms

    return round(output_tokens / (decode_ms / 1000.0), 2)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


def summarize_inference(rows: list[InferenceMetrics]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "samples": 0,
            "ttft_ms": None,
            "tbt_ms": None,
            "tokens_per_sec": None,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
        }

    latest = rows[-1]
    latencies = [row.latency_ms for row in rows]

    return {
        "samples": len(rows),
        "ttft_ms": round(latest.ttft_ms, 2) if latest.ttft_ms is not None else None,
        "tbt_ms": round(latest.tbt_ms, 2) if latest.tbt_ms is not None else None,
        "tokens_per_sec": latest.tokens_per_sec,
        "latency_p50_ms": round(_percentile(latencies, 0.50), 2),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 2),
        "avg_input_tokens": round(sum(row.input_tokens for row in rows) / len(rows), 1),
        "avg_output_tokens": round(sum(row.output_tokens for row in rows) / len(rows), 1),
    }
