from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=128)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    model: str
    latency_ms: float
    estimated_tokens: int
    estimated_cost_usd: float
    tools_used: list[str] = Field(default_factory=list)
    guardrail_blocks: list[str] = Field(default_factory=list)


class SessionResetResponse(BaseModel):
    session_id: str
    cleared: bool


class HealthResponse(BaseModel):
    status: str
    model_id: str
    sessions: int


class CostPricing(BaseModel):
    cpu_hour_usd: float
    tokens_per_char: float
    token_proxy_usd_per_char: float
    deployment: str


class MetricsResponse(BaseModel):
    pricing: CostPricing
    api_cost_consumed_usd: float
    total_requests: int
    estimate_tooltip: str


class InferenceMetricsResponse(BaseModel):
    samples: int
    ttft_ms: float | None = None
    tbt_ms: float | None = None
    tokens_per_sec: float | None = None
    latency_p50_ms: float
    latency_p95_ms: float
    avg_input_tokens: float
    avg_output_tokens: float


class TracesListResponse(BaseModel):
    stats: dict
    traces: list[dict]


class TraceDetailResponse(BaseModel):
    trace_id: str
    spans: list[dict]


class EvalRunRequest(BaseModel):
    benchmark_samples: int = Field(default=10, ge=1, le=20)
    seed: int = Field(default=42)
    assistants: list[str] = Field(default_factory=lambda: ["oss"])


class EvalMetricScore(BaseModel):
    metric: str
    label: str
    percent: float
    total: int


class EvalAssistantResult(BaseModel):
    assistant: str
    model_id: str
    metrics: list[EvalMetricScore]


class EvalRunResponse(BaseModel):
    generated_at: str
    judge_model: str
    results: list[EvalAssistantResult]
    markdown_report: str
