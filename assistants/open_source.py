from assistants.langchain_assistant import LangChainAssistant
from config import OSSConfig
from llm.factory import build_oss_llm
from tools.registry import get_tools


class OpenSourceAssistant(LangChainAssistant):
    """Open-source assistant via LangChain + Hugging Face."""

    def __init__(self, config: OSSConfig, system_prompt: str) -> None:
        super().__init__(
            llm=build_oss_llm(config),
            model_id=config.model_id,
            system_prompt=system_prompt,
            max_history_turns=config.max_history_turns,
            tools=get_tools(),
            session_id="oss",
            guard_identity=True,
        )
