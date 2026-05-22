import logging
from typing import Any

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from memory.conversation import TrimmedChatMessageHistory

logger = logging.getLogger(__name__)


def build_chat_chain(llm: BaseChatModel, system_prompt: str) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_agent_executor(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
    )
    logger.info("Built LangChain agent executor with %d tools", len(tools))
    return executor


class HistoryStore:
    """Session-scoped LangChain message histories."""

    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, TrimmedChatMessageHistory] = {}

    def get(self, session_id: str) -> TrimmedChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = TrimmedChatMessageHistory(
                max_turns=self.max_turns
            )
        return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def clear_all(self) -> None:
        self._sessions.clear()


def invoke_agent(
    executor: AgentExecutor,
    user_input: str,
    history_store: HistoryStore,
    session_id: str,
) -> dict[str, Any]:
    history = history_store.get(session_id)
    return executor.invoke(
        {
            "input": user_input,
            "chat_history": history.messages,
        }
    )
