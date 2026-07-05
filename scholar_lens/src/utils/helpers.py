import argparse
import asyncio
import functools
import re
import time
from collections.abc import Callable
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from ..logger import logger


def parse_quality_score(raw: Any) -> int:
    """Parse a 0-100 evaluator quality score robustly.

    Shared by the review (reflect) and tech-guide (evaluate) score-and-revise
    loops. The prompt asks for a bare integer, but a model may still emit "85",
    "85/100", or "N/A". Prefer a clean integer; otherwise take the FIRST number
    in a leading "<n>/<total>" form (so "85/100" -> 85, not 100); a non-numeric
    value falls back to 0, which forces another attempt rather than crashing.
    """
    text = str(raw if raw is not None else "").strip()
    try:
        return int(text)
    except ValueError:
        pass
    match = re.match(r"-?\d+", text)
    return int(match.group()) if match else 0


def arg_as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("yes", "true", "t", "y", "1"):
            return True
        if value in ("no", "false", "f", "n", "0"):
            return False

    raise argparse.ArgumentTypeError("Boolean value expected")


_AFFIRMATIVE = frozenset({"y", "yes", "yeah", "yep", "true", "t", "1"})
_PLACEHOLDER = frozenset(
    {
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "unnamed",
        "anonymous",
        "not available",
        "empty",
    }
)


def is_affirmative(value: str | None) -> bool:
    """Whether an LLM yes/no field means "yes".

    Tolerant of the common ways a model says yes ("y", "yes", "true", …) instead
    of brittle equality with a single character. Anything else (incl. None) is no.
    """
    return value is not None and value.strip().lower() in _AFFIRMATIVE


def is_placeholder(value: str | None) -> bool:
    """Whether a value is a missing/placeholder marker (N/A, Unknown, None…)."""
    return not value or value.strip().lower() in _PLACEHOLDER


def extract_text_from_html(html_content: str) -> str:
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    for tag_name in ["head", "meta", "script", "style", "title"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    def parse_element(element) -> str:
        if isinstance(element, NavigableString):
            return element.strip()

        if not isinstance(element, Tag):
            return ""

        if element.name == "img":
            alt = element.get("alt", "")
            src = element.get("src", "")
            return f"[Image: alt={alt}, src={src}]"

        if element.name == "a":
            href = element.get("href", "")
            link_text = "".join(parse_element(child) for child in element.children)
            return f"{link_text} ({href})" if href else link_text

        if element.name in ["table", "thead", "tbody", "tr", "td", "th"]:
            content = "".join(parse_element(child) for child in element.children)
            return content

        if element.name in ["code", "pre"]:
            code_content = "".join(parse_element(child) for child in element.children)
            return f"`{code_content}`"

        if element.name == "math":
            math_content = "".join(parse_element(child) for child in element.children)
            return f"$$ {math_content} $$"

        return " ".join(parse_element(child) for child in element.children)

    extracted_text = parse_element(soup)

    # Strip a few LaTeXML extraction artifacts. NOTE: do NOT map "\times" -> "x"
    # here — it corrupts real math (e.g. "a \times b"); math underscores/operators
    # are handled downstream by markdown_math, not by lossy string replacement.
    replacements = {"\\AND": "", "\\n": " ", "footnotemark:": ""}
    for old, new in replacements.items():
        extracted_text = extracted_text.replace(old, new)

    extracted_text = re.sub(r"\s+", " ", extracted_text).strip()

    return extracted_text


def measure_execution_time(func: Callable) -> Callable:
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(
            "'%s' execution time: %.2fs (%.2fmin)",
            func.__name__,
            execution_time,
            execution_time / 60,
        )
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(
            "'%s' execution time: %.2fs (%.2fmin)",
            func.__name__,
            execution_time,
            execution_time / 60,
        )
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
