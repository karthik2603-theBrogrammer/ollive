import logging
from collections.abc import Sequence

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class TrimmedChatMessageHistory(InMemoryChatMessageHistory):
    """LangChain in-memory history with a sliding window."""

    max_turns: int = 10

    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)
        self._trim()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        super().add_messages(messages)
        self._trim()

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            removed = len(self.messages) - max_messages
            self.messages = self.messages[removed:]
            logger.info("Trimmed %d old messages from LangChain history", removed)

    def clear(self) -> None:
        super().clear()
        logger.info("LangChain conversation history cleared")
