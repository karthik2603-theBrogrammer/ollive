import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


@dataclass(frozen=True)
class OSSConfig:
    model_id: str = os.getenv("OSS_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    backend: str = os.getenv("OSS_BACKEND", "local")  # local | api
    hf_token: str = os.getenv("HF_TOKEN", "")
    device: int = int(os.getenv("OSS_DEVICE", "-1"))  # -1 = CPU, 0 = first GPU
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))
    max_tokens: int = int(os.getenv("OSS_MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("OSS_TEMPERATURE", "0.3"))


@dataclass(frozen=True)
class FrontierConfig:
    provider: str = os.getenv("FRONTIER_PROVIDER", "openai")
    model_id: str = os.getenv("FRONTIER_MODEL_ID", "gpt-4o-mini")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))
    max_tokens: int = int(os.getenv("FRONTIER_MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("FRONTIER_TEMPERATURE", "0.7"))


@dataclass(frozen=True)
class JudgeConfig:
    model_id: str = os.getenv("JUDGE_MODEL_ID", "gpt-4o-mini")


@dataclass(frozen=True)
class AppConfig:
    oss: OSSConfig = field(default_factory=OSSConfig)
    frontier: FrontierConfig = field(default_factory=FrontierConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        (
            "You are a helpful personal assistant. "
            "Answer clearly, remember context from the conversation, "
            "and ask clarifying questions when needed."
        ),
    )
    oss_system_prompt: str = os.getenv(
        "OSS_SYSTEM_PROMPT",
        (
            "You are Olive, a local open-source assistant powered by the Qwen model. "
            "Your name is Olive. You are NOT Claude, NOT ChatGPT, and NOT Ollie. "
            "Never say you were made by Anthropic or OpenAI. "
            "If asked who you are, say: 'I'm Olive, an open-source assistant running on Qwen.' "
            "Keep answers concise and stay in character as Olive."
        ),
    )

    @property
    def judge_model_id(self) -> str:
        return self.judge.model_id
