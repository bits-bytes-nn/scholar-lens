"""Markdown links and link lists from titles and URLs."""

from __future__ import annotations

import html
from collections.abc import Iterable
from urllib.parse import quote

from ..text import is_placeholder

# Characters of an untrusted title that kramdown or Liquid would read as
# markup inside "[text](url)": brackets end the link, a | turns the list item
# into a table, braces open Liquid. Each becomes an entity (shown as itself).
_LINK_TEXT_ENTITIES = str.maketrans(
    {"[": "&#91;", "]": "&#93;", "|": "&#124;", "{": "&#123;", "}": "&#125;"}
)


# What a link target keeps as is: RFC 3986 reserved characters and "%" (so an
# encoded URL stays the same). Parentheses, spaces and | are encoded: an
# unbalanced ")" ends the target, a space ends it, a | splits a table cell.
_URL_SAFE = ":/?#@!$&'*+,;=%~"


def md_link_text(text: str) -> str:
    """``text`` (a page or paper title) as plain link text: HTML escaped
    (entities in it decoded first, so ``&amp;`` stays one ampersand) and with
    the characters Markdown or Liquid would read as markup as entities."""
    return html.escape(html.unescape(text), quote=False).translate(_LINK_TEXT_ENTITIES)


def md_link_url(url: str) -> str:
    """``url`` as a Markdown link target: characters that would end the
    target or split a table cell percent-encoded, anything already encoded
    kept."""
    return quote(url, safe=_URL_SAFE)


def link_list(links: Iterable[tuple[str | None, str]]) -> str:
    """``* [title](url)`` lines for ``(title, url)`` pairs, one per URL
    (a trailing slash ignored); a missing or placeholder title, or one that
    is just the URL, shows the URL itself."""
    seen: set[str] = set()
    lines: list[str] = []
    for title, url in links:
        if url.rstrip("/") in seen:
            continue
        seen.add(url.rstrip("/"))
        text = " ".join((title or "").split())
        if is_placeholder(text) or text == url:
            text = url
        lines.append(f"* [{md_link_text(text)}]({md_link_url(url)})")
    return "\n".join(lines)
