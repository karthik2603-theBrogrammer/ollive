from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status

from api.schemas import EvalRunRequest, EvalRunResponse

router = APIRouter(prefix="/v1/eval", tags=["eval"])


def _verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    from api.main import get_api_config

    configured = get_api_config().api_key
    if configured and x_api_key != configured:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _get_eval_service():
    from api.main import get_eval_service

    return get_eval_service()


@router.post("/run", response_model=EvalRunResponse)
def run_eval(
    payload: EvalRunRequest,
    _: None = Depends(_verify_api_key),
    service=Depends(_get_eval_service),
) -> EvalRunResponse:
    assistants = payload.assistants or ["oss"]
    return service.run(
        benchmark_samples=payload.benchmark_samples,
        seed=payload.seed,
        assistants=assistants,
    )
