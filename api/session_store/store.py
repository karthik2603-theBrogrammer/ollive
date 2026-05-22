from __future__ import annotations

from assistants.chain_builder import HistoryStore
from memory.conversation import TrimmedChatMessageHistory


class SessionMemoryStore:
    """API session memory backed by LangChain trimmed chat history."""

    def __init__(self, max_turns: int = 10) -> None:
        self._store = HistoryStore(max_turns=max_turns)

    def get_history(self, session_id: str) -> TrimmedChatMessageHistory:
        return self._store.get(session_id)

    def clear(self, session_id: str) -> None:
        self._store.clear(session_id)

    def clear_all(self) -> None:
        self._store.clear_all()

    def session_count(self) -> int:
        return len(self._store._sessions)
