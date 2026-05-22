from __future__ import annotations

import re
from dataclasses import dataclass

from assistants.response_utils import fix_wrong_identity

OUTPUT_HARM_PATTERNS = [
    r"\bhere(?:'s| is)\s+(?:how|a)\s+(?:to\s+)?(?:make|build|create)\s+(?:a\s+)?(?:bomb|malware)\b",
    r"\bstep\s*1\s*:\s*(?:gather|obtain|buy)\b",
    r"\b(ransomware|keylogger)\s+script\b",
    r"\bpoison\s+(?:someone|them|a person)\b",
]

OUTPUT_REGEX = tuple(re.compile(pattern, re.IGNORECASE) for pattern in OUTPUT_HARM_PATTERNS)

REFUSAL_TEMPLATE = (
    "I can't help with that request. I'm Olive, your open-source assistant, "
    "and I'm designed to stay safe and helpful within appropriate boundaries."
)


@dataclass
class GuardrailResult:
    allowed: bool
    layer: str
    reason: str = ""
    sanitized_text: str = ""


def sanitize_output(text: str, *, guard_identity: bool = True) -> GuardrailResult:
    cleaned = (text or "").strip()
    if not cleaned:
        return GuardrailResult(True, "output_safety", sanitized_text="")

    if guard_identity:
        cleaned = fix_wrong_identity(cleaned)

    for pattern in OUTPUT_REGEX:
        if pattern.search(cleaned):
            return GuardrailResult(
                False,
                "output_safety",
                reason=f"Blocked unsafe model output pattern: {pattern.pattern}",
                sanitized_text=REFUSAL_TEMPLATE,
            )

    return GuardrailResult(True, "output_safety", sanitized_text=cleaned)
