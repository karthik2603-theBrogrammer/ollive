from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from api.observability.metrics import METRICS_STORE
from api.schemas import InferenceMetricsResponse, MetricsResponse

router = APIRouter(prefix="/v1/metrics", tags=["metrics"])


def _verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    from api.main import get_api_config

    configured = get_api_config().api_key
    if configured and x_api_key != configured:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _get_api_config():
    from api.main import get_api_config

    return get_api_config()


@router.get("/cost-latency", response_model=MetricsResponse)
def cost_latency(
    _: None = Depends(_verify_api_key),
    config=Depends(_get_api_config),
) -> MetricsResponse:
    inference = METRICS_STORE.inference_summary()
    fallback_latency = float(inference.get("latency_p50_ms") or 1500.0)
    payload = METRICS_STORE.cost_latency_table(
        cpu_hour_usd=config.cost.cpu_hour_usd,
        tokens_per_char=config.cost.estimated_tokens_per_char,
        fallback_latency_ms=fallback_latency,
    )
    return MetricsResponse(**payload)


@router.get("/inference", response_model=InferenceMetricsResponse)
def inference_metrics(
    _: None = Depends(_verify_api_key),
) -> InferenceMetricsResponse:
    summary = METRICS_STORE.inference_summary()
    return InferenceMetricsResponse(**summary)
