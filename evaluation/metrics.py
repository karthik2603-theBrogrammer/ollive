import logging
from dataclasses import dataclass, field

from assistants.base import AssistantResponse, BaseAssistant

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    user: str
    assistant: str
    latency_ms: float
    model: str
    error: str | None = None


@dataclass
class SessionMetrics:
    name: str
    turns: list[TurnRecord] = field(default_factory=list)

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def successful_turns(self) -> int:
        return sum(1 for t in self.turns if t.error is None)

    @property
    def error_rate(self) -> float:
        if not self.turns:
            return 0.0
        return 1.0 - (self.successful_turns / len(self.turns))

    @property
    def avg_latency_ms(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.latency_ms for t in self.turns) / len(self.turns)

    def record(self, user: str, response: AssistantResponse) -> None:
        self.turns.append(
            TurnRecord(
                user=user,
                assistant=response.text,
                latency_ms=response.latency_ms,
                model=response.model,
                error=response.error,
            )
        )

    def summary(self) -> str:
        return (
            f"**{self.name}**\n"
            f"- Model: `{self.turns[-1].model if self.turns else 'n/a'}`\n"
            f"- Turns: {self.total_turns}\n"
            f"- Success rate: {self.successful_turns}/{self.total_turns}\n"
            f"- Avg latency: {self.avg_latency_ms:.0f} ms\n"
            f"- Error rate: {self.error_rate:.0%}"
        )


class TrackedAssistant:
    """Wraps an assistant to collect lightweight evaluation metrics."""

    def __init__(self, assistant: BaseAssistant, name: str) -> None:
        self.assistant = assistant
        self.metrics = SessionMetrics(name=name)

    def chat(self, user_input: str) -> AssistantResponse:
        response = self.assistant.chat(user_input)
        self.metrics.record(user_input, response)
        return response

    def reset(self) -> None:
        self.assistant.reset()
        self.metrics = SessionMetrics(name=self.metrics.name)
