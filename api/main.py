from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logging_config import configure_logging

configure_logging()

from api.settings import ApiConfig
from api.observability.telemetry import instrument_fastapi, setup_telemetry
from api.routes.chat import router as chat_router
from api.routes.eval import router as eval_router
from api.routes.health import router as health_router
from api.routes.metrics import router as metrics_router
from api.routes.traces import router as traces_router
from api.service.chat_service import ChatService
from api.service.eval_service import EvalService
from config import AppConfig

_api_config: ApiConfig | None = None
_app_config: AppConfig | None = None
_chat_service: ChatService | None = None
_eval_service: EvalService | None = None


def get_api_config() -> ApiConfig:
    assert _api_config is not None
    return _api_config


def get_chat_service() -> ChatService:
    assert _chat_service is not None
    return _chat_service


def get_eval_service() -> EvalService:
    assert _eval_service is not None
    return _eval_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _api_config, _app_config, _chat_service, _eval_service

    _api_config = ApiConfig()
    _app_config = AppConfig()
    setup_telemetry(
        service_name=_api_config.telemetry.service_name,
        enabled=_api_config.telemetry.enabled,
        max_spans=_api_config.telemetry.max_spans,
    )
    _chat_service = ChatService(_api_config, _app_config)
    _eval_service = EvalService(_api_config, _app_config)
    _chat_service.startup()
    instrument_fastapi(app)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="ollive OSS API",
        description=(
            "Public FastAPI deployment for the Olive OSS assistant with guardrails, "
            "memory, tools, OpenTelemetry observability, and official eval endpoints."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    def root_links() -> dict[str, str]:
        return {
            "message": "ollive OSS API",
            "docs": "/docs",
            "health": "/health",
            "observability": "/v1/traces/ui",
            "traces_api": "/v1/traces",
            "traces_stream": "/v1/traces/stream",
            "metrics": "/v1/metrics/cost-latency",
            "inference_metrics": "/v1/metrics/inference",
        }

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(metrics_router)
    app.include_router(traces_router)
    app.include_router(eval_router)
    return app


app = create_app()
