from __future__ import annotations

import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.settings import ApiConfig
from api.guardrails.pipeline import GuardrailPipeline
from api.session_store import SessionMemoryStore
from api.observability.inference_metrics import compute_tokens_per_sec
from api.observability.metrics import METRICS_STORE, RequestMetric, estimate_cost_usd, now_iso
from api.observability.telemetry import get_tracer
from api.schemas import ChatResponse
from api.agent_tools.router import route_tools
from assistants.response_utils import clean_model_response, fix_wrong_identity
from config import AppConfig
from llm.factory import build_oss_llm
from llm.streaming_invoke import count_input_tokens_for_prompt, stream_llm_response
from llm.tokenizer_utils import count_text_tokens, get_llm_tokenizer

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, api_config: ApiConfig, app_config: AppConfig) -> None:
        self.api_config = api_config
        self.app_config = app_config
        self.guardrails = GuardrailPipeline(api_config.guardrails)
        self.memory = SessionMemoryStore(max_turns=api_config.max_history_turns)
        self._llm = None
        self._prompt = None
        self._parser = StrOutputParser()

    def startup(self) -> None:
        if self._llm is not None:
            return
        self._llm = build_oss_llm(self.app_config.oss)
        system_prompt = (
            f"{self.app_config.oss_system_prompt}\n\n"
            "You may receive tool results in the user message. Use them when helpful. "
            "Keep answers concise."
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )
        logger.warning("OSS chat engine loaded model=%s", self.app_config.oss.model_id)

    @property
    def model_id(self) -> str:
        return self.app_config.oss.model_id

    def reset_session(self, session_id: str) -> None:
        self.memory.clear(session_id)

    def _record_inference_span_attrs(self, span, inference) -> None:
        span.set_attribute("inference.latency_ms", inference.latency_ms)
        span.set_attribute("inference.input_tokens", inference.input_tokens)
        span.set_attribute("inference.output_tokens", inference.output_tokens)
        if inference.ttft_ms is not None:
            span.set_attribute("inference.ttft_ms", inference.ttft_ms)
        if inference.tbt_ms is not None:
            span.set_attribute("inference.tbt_ms", inference.tbt_ms)
        if inference.tokens_per_sec is not None:
            span.set_attribute("inference.tokens_per_sec", inference.tokens_per_sec)
        span.set_attribute("inference.stream_chunks", inference.stream_chunks)

    def chat(self, message: str, session_id: str) -> ChatResponse:
        self.startup()
        assert self._llm is not None
        assert self._prompt is not None

        tracer = get_tracer("ollive.api.chat")
        start = time.perf_counter()
        guardrail_blocks: list[str] = []

        with tracer.start_as_current_span("chat.request") as span:
            span.set_attribute("session.id", session_id)
            span.set_attribute("input.length", len(message))

            input_decision = self.guardrails.check_input(message)
            if not input_decision.allowed:
                guardrail_blocks.extend(input_decision.blocked_layers)
                span.set_attribute("guardrail.blocks", ",".join(guardrail_blocks))
                span.set_attribute("guardrail.blocked", True)
                latency_ms = (time.perf_counter() - start) * 1000
                tokens, cost = estimate_cost_usd(
                    latency_ms,
                    len(message),
                    self.api_config.cost.cpu_hour_usd,
                    self.api_config.cost.estimated_tokens_per_char,
                )
                METRICS_STORE.record(
                    RequestMetric(
                        timestamp=now_iso(),
                        session_id=session_id,
                        latency_ms=latency_ms,
                        input_chars=len(message),
                        output_chars=len(input_decision.response_text),
                        estimated_tokens=tokens,
                        estimated_cost_usd=cost,
                        guardrail_blocks=guardrail_blocks,
                    )
                )
                return ChatResponse(
                    session_id=session_id,
                    response=input_decision.response_text,
                    model=self.model_id,
                    latency_ms=latency_ms,
                    estimated_tokens=tokens,
                    estimated_cost_usd=cost,
                    tools_used=[],
                    guardrail_blocks=guardrail_blocks,
                )

            tool_route = route_tools(message)
            model_input = f"{tool_route.context_prefix}{message}" if tool_route.context_prefix else message

            history = self.memory.get_history(session_id)
            input_tokens = count_input_tokens_for_prompt(
                self._llm,
                self._prompt,
                model_input=model_input,
                history=history.messages,
            )
            prompt_messages = self._prompt.format_messages(
                input=model_input,
                history=history.messages,
            )
            raw, inference = stream_llm_response(
                self._llm,
                prompt_messages,
                input_tokens=input_tokens,
            )
            text = clean_model_response(self._parser.invoke(raw))
            if not text and raw:
                text = str(raw).strip()
            text = fix_wrong_identity(text)
            tokenizer = get_llm_tokenizer(self._llm)
            final_output_tokens = count_text_tokens(tokenizer, text)
            if final_output_tokens != inference.output_tokens:
                inference = replace(
                    inference,
                    output_tokens=final_output_tokens,
                    tokens_per_sec=compute_tokens_per_sec(
                        final_output_tokens,
                        inference.latency_ms,
                        inference.ttft_ms,
                    ),
                )

            output_decision = self.guardrails.check_output(text)
            if not output_decision.allowed:
                guardrail_blocks.extend(output_decision.blocked_layers)
                text = output_decision.response_text

            history.add_message(HumanMessage(content=message))
            history.add_message(AIMessage(content=text))

            latency_ms = (time.perf_counter() - start) * 1000
            total_chars = len(message) + len(text)
            tokens, cost = estimate_cost_usd(
                latency_ms,
                total_chars,
                self.api_config.cost.cpu_hour_usd,
                self.api_config.cost.estimated_tokens_per_char,
            )

            METRICS_STORE.record(
                RequestMetric(
                    timestamp=now_iso(),
                    session_id=session_id,
                    latency_ms=latency_ms,
                    input_chars=len(message),
                    output_chars=len(text),
                    estimated_tokens=tokens,
                    estimated_cost_usd=cost,
                    guardrail_blocks=guardrail_blocks,
                    tools_used=tool_route.tools_used,
                    inference_latency_ms=inference.latency_ms,
                    ttft_ms=inference.ttft_ms,
                    tbt_ms=inference.tbt_ms,
                    input_tokens=inference.input_tokens,
                    output_tokens=inference.output_tokens,
                    tokens_per_sec=inference.tokens_per_sec,
                )
            )

            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("tools.count", len(tool_route.tools_used))
            span.set_attribute("output.length", len(text))
            self._record_inference_span_attrs(span, inference)
            if guardrail_blocks:
                span.set_attribute("guardrail.blocks", ",".join(guardrail_blocks))
                span.set_attribute("guardrail.blocked", True)
            if tool_route.tools_used:
                span.set_attribute("tools.used", ",".join(tool_route.tools_used))

            return ChatResponse(
                session_id=session_id,
                response=text,
                model=self.model_id,
                latency_ms=round(latency_ms, 1),
                estimated_tokens=tokens,
                estimated_cost_usd=cost,
                tools_used=tool_route.tools_used,
                guardrail_blocks=guardrail_blocks,
            )
