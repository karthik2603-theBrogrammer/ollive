from __future__ import annotations

import re
from dataclasses import dataclass, field

HARMFUL_PATTERNS = [
    r"\b(make|build|write|create)\s+(a\s+)?(bomb|malware|ransomware|virus)\b",
    r"\bhow\s+to\s+(kill|poison|murder|harm)\b",
    r"\b(synthesize|manufacture)\s+(meth|drugs|explosives)\b",
    r"\bsteal\s+(someone'?s?\s+)?identity\b",
    r"\b(child\s+porn|csam)\b",
    r"\bjailbreak\b",
    r"\bignore\s+(all\s+)?(previous|prior)\s+instructions\b",
    r"\bpretend\s+you\s+are\s+dan\b",
    r"\bno\s+safety\s+(rules|filters|restrictions)\b",
]

BLOCKLIST = tuple(re.compile(pattern, re.IGNORECASE) for pattern in HARMFUL_PATTERNS)


@dataclass
class GuardrailResult:
    allowed: bool
    layer: str
    reason: str = ""


def check_input_safety(text: str) -> GuardrailResult:
    cleaned = (text or "").strip()
    if not cleaned:
        return GuardrailResult(False, "input_safety", "Empty input")

    for pattern in BLOCKLIST:
        if pattern.search(cleaned):
            return GuardrailResult(
                False,
                "input_safety",
                f"Blocked harmful or jailbreak pattern: {pattern.pattern}",
            )
    return GuardrailResult(True, "input_safety")
