from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate


def get_llm_tokenizer(llm: BaseChatModel):
    inner = getattr(llm, "llm", None)
    pipeline = getattr(inner, "pipeline", None)
    if pipeline is not None:
        return getattr(pipeline, "tokenizer", None)
    return None


def count_text_tokens(tokenizer, text: str) -> int:
    if not text or tokenizer is None:
        return 0
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)


def count_prompt_tokens(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    *,
    model_input: str,
    history: list[BaseMessage],
) -> int:
    tokenizer = get_llm_tokenizer(llm)
    if tokenizer is None:
        return 0
    messages = prompt.format_messages(input=model_input, history=history)
    parts: list[str] = []
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        parts.append(f"{role}: {message.content}")
    return count_text_tokens(tokenizer, "\n".join(parts))
