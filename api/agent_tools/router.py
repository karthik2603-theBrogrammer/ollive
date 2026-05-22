from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from api.agent_tools.registry import (
    calculator,
    fetch_wikipedia_summary,
    get_current_time,
    unit_converter,
    word_counter,
)

logger = logging.getLogger(__name__)

_MATH_PATTERN = re.compile(r"(?:\d+\s*[\+\-\*/\(\)]\s*)+\d+")
_CONVERT_PATTERN = re.compile(
    r"convert\s+(?P<value>\d+(?:\.\d+)?)\s+(?P<from_unit>[a-zA-Z]+)\s+(?:to|into)\s+(?P<to_unit>[a-zA-Z]+)",
    re.IGNORECASE,
)
_TIME_PATTERN = re.compile(
    r"\b("
    r"what(?:'s| is) the (?:current )?time|"
    r"what time is it|"
    r"current time|"
    r"what(?:'s| is) today'?s date|"
    r"what date is it|"
    r"time now"
    r")\b",
    re.IGNORECASE,
)
_WORD_COUNT_PATTERN = re.compile(
    r"\b(count words|word count|how many words)\b",
    re.IGNORECASE,
)
_WIKI_PREFIXES = (
    "tell me about",
    "tell me more about",
    "who is",
    "who was",
    "summary of",
    "wikipedia summary of",
    "wikipedia page on",
    "wikipedia",
    "wiki about",
    "look up",
    "lookup",
    "search wikipedia for",
)


@dataclass
class ToolRouteResult:
    tools_used: list[str]
    context_prefix: str


def _looks_like_math(text: str) -> bool:
    return bool(_MATH_PATTERN.search(text) and re.search(r"[\+\-\*/\(\)]", text))


def _extract_calculator_expression(text: str) -> str | None:
    if not _looks_like_math(text):
        return None
    lower = text.lower()
    if not any(token in lower for token in ("calculate", "compute", "evaluate", "solve", "what is", "what's")):
        return None
    match = _MATH_PATTERN.search(text)
    return match.group(0).strip() if match else None


def _extract_wikipedia_query(text: str) -> str | None:
    cleaned = text.strip()
    lower = cleaned.lower()

    if _TIME_PATTERN.search(lower) or _CONVERT_PATTERN.search(cleaned) or _WORD_COUNT_PATTERN.search(lower):
        return None
    if _looks_like_math(cleaned):
        return None

    for prefix in _WIKI_PREFIXES:
        if lower.startswith(prefix):
            query = cleaned[len(prefix) :].strip(" ?.:,-")
            return query or None

    what_match = re.match(r"what(?:'s| is| was)\s+(?P<query>.+?)\??$", cleaned, re.IGNORECASE)
    if what_match:
        query = what_match.group("query").strip(" ?.")
        if query and not re.search(r"\d", query):
            return query

    if lower.startswith("define "):
        return cleaned[7:].strip(" ?.")

    return None


def _extract_word_count_text(text: str) -> str:
    if ":" in text:
        return text.split(":", 1)[1].strip()
    match = _WORD_COUNT_PATTERN.search(text)
    if match:
        remainder = text[match.end() :].strip(" :in-for-of")
        if remainder:
            return remainder
    return text


def route_tools(user_input: str) -> ToolRouteResult:
    """Heuristic tool router for small OSS models without reliable native tool-calling."""
    text = user_input.strip()
    if not text:
        return ToolRouteResult(tools_used=[], context_prefix="")

    used: list[str] = []
    snippets: list[str] = []

    if _TIME_PATTERN.search(text):
        result = get_current_time.invoke({})
        used.append("get_current_time")
        snippets.append(f"[tool:get_current_time] {result}")

    if _WORD_COUNT_PATTERN.search(text):
        payload = _extract_word_count_text(text)
        result = word_counter.invoke({"text": payload})
        used.append("word_counter")
        snippets.append(f"[tool:word_counter] {result}")

    convert_match = _CONVERT_PATTERN.search(text)
    if convert_match:
        result = unit_converter.invoke(
            {
                "value": float(convert_match.group("value")),
                "from_unit": convert_match.group("from_unit"),
                "to_unit": convert_match.group("to_unit"),
            }
        )
        used.append("unit_converter")
        snippets.append(f"[tool:unit_converter] {result}")

    expression = _extract_calculator_expression(text)
    if expression:
        try:
            result = calculator.invoke({"expression": expression})
            used.append("calculator")
            snippets.append(f"[tool:calculator] {expression} = {result}")
        except Exception as exc:
            logger.warning("Calculator failed for expression=%r: %s", expression, exc)

    wiki_query = _extract_wikipedia_query(text)
    if wiki_query:
        try:
            result = fetch_wikipedia_summary(wiki_query)
            used.append("wikipedia_summary")
            snippets.append(f"[tool:wikipedia_summary] {result}")
        except Exception as exc:
            logger.warning("Wikipedia routing failed for query=%r: %s", wiki_query, exc)
            used.append("wikipedia_summary")
            snippets.append(f"[tool:wikipedia_summary] Wikipedia lookup failed: {exc}")

    if not snippets:
        return ToolRouteResult(tools_used=[], context_prefix="")

    prefix = (
        "Tool results (use these facts directly in your answer; do not invent values):\n"
        + "\n".join(snippets)
        + "\n\nUser message: "
    )
    return ToolRouteResult(tools_used=used, context_prefix=prefix)
