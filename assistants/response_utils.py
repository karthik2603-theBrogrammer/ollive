import re

IM_END = "<|" + "im_end" + "|>"

_IM_START = re.compile(r"<\|im_start\|>\s*", re.IGNORECASE)
_IM_END = re.compile(r"<\|im_end\|>", re.IGNORECASE)
_ASSISTANT_BLOCK = re.compile(
    r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)",
    re.DOTALL | re.IGNORECASE,
)
_WRONG_IDENTITY = re.compile(
    r"\b(i am|i'm|my name is|call me)\s+(claude|chatgpt|gpt|ollie)\b",
    re.IGNORECASE,
)
_ANTHROPIC_CLAIM = re.compile(r"\banthropic\b", re.IGNORECASE)

OLIVE_IDENTITY_REPLY = (
    "I'm Olive, a local open-source assistant running on Qwen. How can I help you?"
)


def clean_model_response(text: str) -> str:
    """Keep only the assistant reply when chat-template tokens leak through."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    if "<|im_start|>" not in cleaned.lower() and IM_END not in cleaned:
        return cleaned

    match = _ASSISTANT_BLOCK.search(cleaned)
    if match:
        return match.group(1).strip()

    cleaned = _IM_END.sub("", cleaned)
    cleaned = _IM_START.sub("", cleaned)
    for role in ("system", "user", "assistant"):
        cleaned = re.sub(rf"^{role}\s*", "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def fix_wrong_identity(text: str) -> str:
    """
    Small OSS models often hallucinate Claude/ChatGPT personas.
    Rewrite obvious wrong-identity intros as Olive.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    lower = cleaned.lower()
    wrong_name = _WRONG_IDENTITY.search(cleaned) is not None
    anthropic_claim = _ANTHROPIC_CLAIM.search(cleaned) is not None
    mentions_claude = "claude" in lower

    if wrong_name or anthropic_claim or (
        mentions_claude and any(p in lower for p in ("i am", "i'm", "my name", "developed by"))
    ):
        return OLIVE_IDENTITY_REPLY

    return cleaned
