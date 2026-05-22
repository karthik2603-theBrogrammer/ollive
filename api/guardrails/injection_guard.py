from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"<\s*script",
    r"system\s*:\s*",
    r"<\|im_start\|>",
    r"###\s*instruction",
    r"override\s+safety",
    r"developer\s+mode",
]

INJECTION_REGEX = tuple(re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS)


@dataclass
class GuardrailResult:
    allowed: bool
    layer: str
    reason: str = ""


def check_injection_and_limits(text: str, *, max_chars: int) -> GuardrailResult:
    cleaned = (text or "").strip()
    if len(cleaned) > max_chars:
        return GuardrailResult(
            False,
            "injection_guard",
            f"Input exceeds max length ({len(cleaned)} > {max_chars})",
        )

    for pattern in INJECTION_REGEX:
        if pattern.search(cleaned):
            return GuardrailResult(
                False,
                "injection_guard",
                f"Blocked suspicious injection pattern: {pattern.pattern}",
            )
    return GuardrailResult(True, "injection_guard")
