from assistants.langchain_assistant import LangChainAssistant
from config import FrontierConfig
from llm.factory import build_frontier_llm
from tools.registry import get_tools


class FrontierAssistant(LangChainAssistant):
    """Frontier model assistant via LangChain + hosted API."""

    def __init__(self, config: FrontierConfig, system_prompt: str) -> None:
        super().__init__(
            llm=build_frontier_llm(config),
            model_id=config.model_id,
            system_prompt=system_prompt,
            max_history_turns=config.max_history_turns,
            tools=get_tools(),
            session_id="frontier",
        )
