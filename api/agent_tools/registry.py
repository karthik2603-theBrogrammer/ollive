from __future__ import annotations

import ast
import json
import logging
import operator
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_USER_AGENT = "ollive-api/1.0 (local-assignment; python-urllib)"
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

_UNIT_ALIASES = {
    "c": "c",
    "celsius": "c",
    "f": "f",
    "fahrenheit": "f",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "mi": "mi",
    "mile": "mi",
    "miles": "mi",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent too large")
        return _SAFE_OPERATORS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")


def _wiki_request(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _wiki_search_title(query: str) -> str | None:
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
            "utf8": 1,
        }
    )
    payload = _wiki_request(f"{_WIKI_SEARCH_URL}?{params}")
    hits = payload.get("query", {}).get("search", [])
    if not hits:
        return None
    return hits[0].get("title")


def _wiki_summary_for_title(title: str) -> str:
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    payload = _wiki_request(_WIKI_SUMMARY_URL.format(title=encoded))
    if payload.get("type") == "disambiguation":
        raise ValueError(
            f"'{title}' matches multiple Wikipedia pages. Ask with a more specific title."
        )
    extract = payload.get("extract", "").strip()
    if not extract:
        raise ValueError(f"No summary text found for '{title}'.")
    page_title = payload.get("title", title)
    return f"{page_title}: {extract}"


def fetch_wikipedia_summary(query: str) -> str:
    """Fetch a Wikipedia summary using the public REST API with search fallback."""
    cleaned = query.strip().strip("?.!")
    if not cleaned:
        raise ValueError("Wikipedia query is empty.")

    try:
        return _wiki_summary_for_title(cleaned)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    except ValueError:
        raise

    resolved = _wiki_search_title(cleaned)
    if not resolved:
        raise ValueError(f"No Wikipedia article found for '{cleaned}'.")
    return _wiki_summary_for_title(resolved)


@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression with +, -, *, /, parentheses, and powers.

    Use when the user asks to calculate, compute, evaluate, or solve a numeric expression.
    Examples: "(17 * 23) + 4", "2 ** 10", "100 / 4".
    """
    tree = ast.parse(expression.strip(), mode="eval")
    result = _safe_eval(tree.body)
    if result.is_integer():
        return str(int(result))
    return str(round(result, 6))


@tool("get_current_time")
def get_current_time(timezone_name: str = "UTC") -> str:
    """Return the current date and time in UTC.

    Use when the user asks for the current time, today's date, or what time it is now.
    The timezone_name argument is accepted for compatibility but UTC is returned.
    """
    _ = timezone_name
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


@tool("word_counter")
def word_counter(text: str) -> str:
    """Count words and characters in a piece of text.

    Use when the user asks for a word count, character count, or how long a passage is.
    Pass the exact text to analyze, not the full user question.
    """
    words = len(re.findall(r"\b\w+\b", text))
    chars = len(text)
    return f"words={words}, characters={chars}"


@tool("wikipedia_summary")
def wikipedia_summary(query: str) -> str:
    """Look up a short Wikipedia summary for a person, place, concept, or topic.

    Use for factual background questions such as "Who is Marie Curie?",
    "Tell me about the Eiffel Tower", or "Wikipedia summary of photosynthesis".
    Pass a concise topic name or article title, not the full conversational question.
    """
    try:
        return fetch_wikipedia_summary(query)
    except Exception as exc:
        logger.warning("Wikipedia lookup failed for query=%r: %s", query, exc)
        return f"Wikipedia lookup failed: {exc}"


@tool("unit_converter")
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numeric value between supported units.

    Supported pairs: Celsius/Fahrenheit, kilometers/miles, kilograms/pounds, meters/feet.
    Use when the user asks to convert measurements, for example "convert 10 km to miles".
    """
    from_key = _UNIT_ALIASES.get(from_unit.strip().lower())
    to_key = _UNIT_ALIASES.get(to_unit.strip().lower())
    if not from_key or not to_key:
        supported = ", ".join(sorted(set(_UNIT_ALIASES)))
        return f"Unsupported units. Supported aliases include: {supported}"

    conversions = {
        ("c", "f"): lambda v: (v * 9 / 5) + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
        ("km", "mi"): lambda v: v * 0.621371,
        ("mi", "km"): lambda v: v / 0.621371,
        ("kg", "lb"): lambda v: v * 2.20462,
        ("lb", "kg"): lambda v: v / 2.20462,
        ("m", "ft"): lambda v: v * 3.28084,
        ("ft", "m"): lambda v: v / 3.28084,
    }
    key = (from_key, to_key)
    if key not in conversions:
        return f"Unsupported conversion: {from_unit} -> {to_unit}"
    converted = conversions[key](value)
    return f"{value} {from_unit} = {round(converted, 4)} {to_unit}"


def get_api_tools():
    return [calculator, get_current_time, word_counter, wikipedia_summary, unit_converter]
