from __future__ import annotations

from typing import Any

TOKEN_PROXY_USD_PER_CHAR: float = 0.000001


def runtime_cost_usd(latency_ms: float, cpu_hour_usd: float) -> float:
    return (latency_ms / 3_600_000) * cpu_hour_usd


def token_proxy_cost_usd(total_chars: int, tokens_per_char: float) -> float:
    return max(total_chars, 1) * tokens_per_char * TOKEN_PROXY_USD_PER_CHAR


def estimate_request_cost_usd(
    *,
    latency_ms: float,
    total_chars: int,
    cpu_hour_usd: float,
    tokens_per_char: float,
) -> tuple[int, float]:
    runtime = runtime_cost_usd(latency_ms, cpu_hour_usd)
    token = token_proxy_cost_usd(total_chars, tokens_per_char)
    estimated_tokens = int(max(total_chars, 1) * tokens_per_char)
    return estimated_tokens, round(runtime + token, 8)


def pricing_summary(*, cpu_hour_usd: float, tokens_per_char: float) -> dict[str, float | str]:
    return {
        "cpu_hour_usd": cpu_hour_usd,
        "tokens_per_char": tokens_per_char,
        "token_proxy_usd_per_char": TOKEN_PROXY_USD_PER_CHAR,
        "deployment": "oss_cpu",
    }


def consumed_cost_tooltip(
    *,
    cpu_hour_usd: float,
    tokens_per_char: float,
    total_cost_usd: float,
    total_requests: int,
) -> str:
    if total_requests <= 0:
        return (
            f"No requests recorded yet. Each request is estimated as runtime + token proxy using "
            f"${cpu_hour_usd:.4f}/CPU-hour and {tokens_per_char:.2f} tok/char × "
            f"${TOKEN_PROXY_USD_PER_CHAR:.6f}/char."
        )

    avg_cost = total_cost_usd / total_requests
    return (
        f"API $ consumed = sum of estimated cost across {total_requests} request(s) = "
        f"${total_cost_usd:.8f}. Per request: runtime = (latency_ms ÷ 3,600,000) × "
        f"${cpu_hour_usd:.4f}/hr; token proxy = (input + output chars) × {tokens_per_char:.2f} × "
        f"${TOKEN_PROXY_USD_PER_CHAR:.6f}. Request cost = runtime + token proxy. "
        f"Average per request: ${avg_cost:.8f}."
    )


def build_cost_latency_table(
    request_rows: list[Any],
    *,
    cpu_hour_usd: float,
    tokens_per_char: float,
    fallback_latency_ms: float = 1500.0,
) -> dict[str, Any]:
    del fallback_latency_ms  # kept for caller compatibility

    total_cost = sum(row.estimated_cost_usd for row in request_rows)
    total_requests = len(request_rows)

    return {
        "pricing": pricing_summary(cpu_hour_usd=cpu_hour_usd, tokens_per_char=tokens_per_char),
        "api_cost_consumed_usd": round(total_cost, 8),
        "total_requests": total_requests,
        "estimate_tooltip": consumed_cost_tooltip(
            cpu_hour_usd=cpu_hour_usd,
            tokens_per_char=tokens_per_char,
            total_cost_usd=total_cost,
            total_requests=total_requests,
        ),
    }
