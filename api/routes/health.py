from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


def _get_chat_service():
    from api.main import get_chat_service

    return get_chat_service()


@router.get("/health", response_model=HealthResponse)
def health(service=Depends(_get_chat_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_id=service.model_id,
        sessions=service.memory.session_count(),
    )
