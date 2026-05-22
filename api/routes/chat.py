from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from api.schemas import ChatRequest, ChatResponse, SessionResetResponse

router = APIRouter(prefix="/v1", tags=["chat"])


def _get_chat_service():
    from api.main import get_chat_service

    return get_chat_service()


def _verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    from api.main import get_api_config

    configured = get_api_config().api_key
    if configured and x_api_key != configured:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    _: None = Depends(_verify_api_key),
    service=Depends(_get_chat_service),
) -> ChatResponse:
    return service.chat(payload.message, payload.session_id)


@router.delete("/sessions/{session_id}", response_model=SessionResetResponse)
def reset_session(
    session_id: str,
    _: None = Depends(_verify_api_key),
    service=Depends(_get_chat_service),
) -> SessionResetResponse:
    service.reset_session(session_id)
    return SessionResetResponse(session_id=session_id, cleared=True)
