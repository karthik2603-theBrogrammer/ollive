from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage

from api.observability.inference_metrics import InferenceMetrics, compute_tokens_per_sec
from llm.tokenizer_utils import count_prompt_tokens, count_text_tokens, get_llm_tokenizer


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, AIMessageChunk):
        content = chunk.content
        return content if isinstance(content, str) else str(content or "")
    if hasattr(chunk, "content"):
        content = chunk.content
        return content if isinstance(content, str) else str(content or "")
    return str(chunk)


def stream_llm_response(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    *,
    input_tokens: int,
) -> tuple[str, InferenceMetrics]:
    """Stream model output and compute TTFT/TBT from real chunk arrival times."""
    inference_start = time.perf_counter()
    chunk_times: list[float] = []
    parts: list[str] = []

    try:
        stream = llm.stream(messages)
    except (NotImplementedError, TypeError, AttributeError):
        stream = None

    if stream is not None:
        for chunk in stream:
            text = _chunk_text(chunk)
            if not text:
                continue
            chunk_times.append(time.perf_counter())
            parts.append(text)

    if not parts:
        batch_start = time.perf_counter()
        batch = llm.invoke(messages)
        response_text = _chunk_text(batch)
        inference_end = time.perf_counter()
        latency_ms = (inference_end - batch_start) * 1000
        tokenizer = get_llm_tokenizer(llm)
        output_tokens = count_text_tokens(tokenizer, response_text)
        latency_rounded = round(latency_ms, 2)
        ttft_rounded = round(latency_ms, 2)
        return response_text, InferenceMetrics(
            latency_ms=latency_rounded,
            ttft_ms=ttft_rounded,
            tbt_ms=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stream_chunks=1 if response_text else 0,
            tokens_per_sec=compute_tokens_per_sec(output_tokens, latency_rounded, ttft_rounded),
        )

    inference_end = time.perf_counter()
    response_text = "".join(parts)
    latency_ms = (inference_end - inference_start) * 1000

    ttft_ms: float | None = None
    tbt_ms: float | None = None
    if chunk_times:
        ttft_ms = (chunk_times[0] - inference_start) * 1000
    if len(chunk_times) >= 2:
        gaps_ms = [(chunk_times[i] - chunk_times[i - 1]) * 1000 for i in range(1, len(chunk_times))]
        tbt_ms = sum(gaps_ms) / len(gaps_ms)

    tokenizer = get_llm_tokenizer(llm)
    output_tokens = count_text_tokens(tokenizer, response_text)
    latency_rounded = round(latency_ms, 2)
    ttft_rounded = round(ttft_ms, 2) if ttft_ms is not None else None
    tbt_rounded = round(tbt_ms, 2) if tbt_ms is not None else None

    return response_text, InferenceMetrics(
        latency_ms=latency_rounded,
        ttft_ms=ttft_rounded,
        tbt_ms=tbt_rounded,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stream_chunks=len(chunk_times),
        tokens_per_sec=compute_tokens_per_sec(output_tokens, latency_rounded, ttft_rounded),
    )


def count_input_tokens_for_prompt(
    llm: BaseChatModel,
    prompt,
    *,
    model_input: str,
    history: list[BaseMessage],
) -> int:
    return count_prompt_tokens(llm, prompt, model_input=model_input, history=history)
