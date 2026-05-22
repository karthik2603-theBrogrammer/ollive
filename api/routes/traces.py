from __future__ import annotations

import asyncio
import json
import queue

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, StreamingResponse

from api.observability.trace_store import TRACE_STORE
from api.observability.trace_ui import render_trace_dashboard
from api.schemas import TraceDetailResponse, TracesListResponse

router = APIRouter(prefix="/v1/traces", tags=["traces"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _get_api_config():
    from api.main import get_api_config

    return get_api_config()


def _sse_message(*, event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("", response_model=TracesListResponse)
def list_traces(
    limit: int = Query(default=40, ge=1, le=200),
    _config=Depends(_get_api_config),
) -> TracesListResponse:
    payload = TRACE_STORE.build_payload(limit=limit)
    return TracesListResponse(stats=payload["stats"], traces=payload["traces"])


@router.get("/ui", response_class=HTMLResponse)
def traces_ui(_config=Depends(_get_api_config)) -> HTMLResponse:
    config = _config
    payload = TRACE_STORE.build_payload(limit=40)
    html = render_trace_dashboard(
        service_name=config.telemetry.service_name,
        stats=payload["stats"],
        traces=payload["traces"],
    )
    return HTMLResponse(content=html)


@router.get("/stream")
async def trace_stream(
    limit: int = Query(default=40, ge=1, le=200),
    _config=Depends(_get_api_config),
) -> StreamingResponse:
    async def event_generator():
        listener = TRACE_STORE.subscribe()
        try:
            yield _sse_message(event="snapshot", data=TRACE_STORE.build_payload(limit=limit))
            while True:
                try:
                    await asyncio.to_thread(listener.get, True, 30.0)
                except queue.Empty:
                    yield ": heartbeat\n\n"
                    continue
                yield _sse_message(event="update", data=TRACE_STORE.build_payload(limit=limit))
        finally:
            TRACE_STORE.unsubscribe(listener)

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.get("/{trace_id}", response_model=TraceDetailResponse)
def get_trace(trace_id: str, _config=Depends(_get_api_config)) -> TraceDetailResponse:
    spans = TRACE_STORE.by_trace_id(trace_id)
    if not spans:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace not found")
    return TraceDetailResponse(
        trace_id=spans[0].trace_id,
        spans=[span.to_dict() for span in spans],
    )
