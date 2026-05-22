import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_classic.agents import AgentExecutor
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from assistants.chain_builder import (
    HistoryStore,
    build_agent_executor,
    build_chat_chain,
    invoke_agent,
)
from assistants.response_utils import clean_model_response, fix_wrong_identity

logger = logging.getLogger(__name__)


@dataclass
class AssistantResponse:
    text: str
    latency_ms: float
    model: str
    error: str | None = None


class BaseAssistant(ABC):
    @abstractmethod
    def chat(self, user_input: str) -> AssistantResponse:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class LangChainAssistant(BaseAssistant):
    """
    LangChain-powered assistant with short-term memory.

    Uses a simple LCEL chat chain when no tools are registered.
    Switches to a tool-calling AgentExecutor automatically once tools are added.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        model_id: str,
        system_prompt: str,
        max_history_turns: int = 10,
        tools: list[BaseTool] | None = None,
        session_id: str = "default",
        guard_identity: bool = False,
    ) -> None:
        self.llm = llm
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.tools = tools or []
        self.guard_identity = guard_identity

        self.history_store = HistoryStore(max_turns=max_history_turns)
        self._agent_executor: AgentExecutor | None = None
        self._chat_chain: Runnable | None = None

        if self.tools:
            self._agent_executor = build_agent_executor(
                llm=self.llm,
                tools=self.tools,
                system_prompt=self.system_prompt,
            )
        else:
            self._chat_chain = build_chat_chain(self.llm, self.system_prompt)

    def chat(self, user_input: str) -> AssistantResponse:
        user_input = user_input.strip()
        if not user_input:
            return AssistantResponse(
                text="Please enter a message.",
                latency_ms=0.0,
                model=self.model_id,
                error="empty_input",
            )

        start = time.perf_counter()
        try:
            text = self._invoke(user_input)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "LangChain chat completed model=%s latency_ms=%.1f tools=%d",
                self.model_id,
                elapsed_ms,
                len(self.tools),
            )
            return AssistantResponse(
                text=text.strip(),
                latency_ms=elapsed_ms,
                model=self.model_id,
            )
        except Exception as exc:
            logger.exception("LangChain assistant failed")
            return AssistantResponse(
                text=f"Sorry, something went wrong: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
                model=self.model_id,
                error=str(exc),
            )

    def reset(self) -> None:
        self.history_store.clear(self.session_id)

    def _invoke(self, user_input: str) -> str:
        if self._agent_executor is not None:
            result = invoke_agent(
                executor=self._agent_executor,
                user_input=user_input,
                history_store=self.history_store,
                session_id=self.session_id,
            )
            text = clean_model_response(str(result.get("output", "")))
            if self.guard_identity:
                text = fix_wrong_identity(text)
            history = self.history_store.get(self.session_id)
            history.add_message(HumanMessage(content=user_input))
            history.add_message(AIMessage(content=text))
            return text

        assert self._chat_chain is not None
        history = self.history_store.get(self.session_id)
        raw = self._chat_chain.invoke(
            {
                "input": user_input,
                "history": history.messages,
            }
        )
        text = clean_model_response(str(raw))
        if not text and raw:
            text = str(raw).strip()
        if self.guard_identity:
            text = fix_wrong_identity(text)
        history.add_message(HumanMessage(content=user_input))
        history.add_message(AIMessage(content=text))
        return text
