"""Markdown links: link lists from titles and URLs, one parser of the
links a post has, and the check that a post links only its sources."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, unquote, urlsplit

from scholar_lens.domain.ids import paper_link_key
from scholar_lens.domain.markdown.math import in_spans, protected_spans
from scholar_lens.domain.text import is_placeholder

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


# ---- reading links -------------------------------------------------------------

# Link text with brackets nested two levels deep (a linked image's alt).
_TEXT = r"(?:[^\[\]\n]|\[(?:[^\[\]\n]|\[[^\[\]\n]*\])*\])*"
# A destination: in angle brackets, or without spaces and with balanced
# parentheses (two levels).
_DESTINATION = r"<[^<>\n]*>|(?:[^\s()<>]|\((?:[^\s()]|\([^\s()]*\))*\))+"
_TITLE = r"""(?:[ \t]+(?:"[^"\n]*"|'[^'\n]*'|\([^()\n]*\)))?"""
# One pass over the prose: inline links, full and collapsed references,
# shortcut references (no nested brackets) and autolinks. An image
# (``![alt](src)``) and an escaped bracket open none.
_LINK = re.compile(
    rf"(?<![!\\])\[(?P<text>{_TEXT})\]"
    rf"(?:\([ \t]*(?P<dest>{_DESTINATION})?{_TITLE}[ \t]*\)"
    r"|\[(?P<label>[^\[\]\n]*)\])"
    r"|(?<![!\\\]])\[(?P<short>[^\[\]\n]+)\](?![(\[:])"
    r"|<(?P<auto>[A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\s]+)>"
)
# A reference definition (not a footnote's ``[^1]:``), on its own line.
_DEFINITION = re.compile(
    r"^[ ]{0,3}\[(?!\^)(?P<label>[^\[\]\n]+)\]:[ \t]*(?P<dest><[^<>\n]*>|\S+)"
    rf"{_TITLE}[ \t]*(?:\n|$)",
    re.MULTILINE,
)
# A target that leaves the site: a scheme with "//" or protocol-relative.
_ABSOLUTE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:)?//")

LinkForm = Literal["inline", "reference", "autolink", "definition"]


@dataclass(frozen=True)
class MarkdownLink:
    """One link of a Markdown text: its span, its text (an autolink's is its
    URL, a definition's its label), its target (a reference's from its
    definition) and its form."""

    start: int
    end: int
    text: str
    target: str
    form: LinkForm
    # A reference's or definition's label, as Markdown compares them.
    label: str = ""


def _label(text: str) -> str:
    return " ".join(text.split()).casefold()


def _destination(value: str) -> str:
    value = value.strip()
    return value[1:-1].strip() if value.startswith("<") else value


def markdown_links(markdown: str) -> list[MarkdownLink]:
    """Every link of ``markdown`` outside code and math, in order: inline
    links (``[t](url "title")``, nested brackets in the text), references
    to a definition (``[t][label]``, ``[label][]``, ``[label]``), autolinks
    (``<url>``) and the definitions themselves. Images are not links here;
    a reference without a definition is plain text."""
    protected = in_spans(protected_spans(markdown))
    definitions: dict[str, str] = {}
    links: list[MarkdownLink] = []
    defined_spans: list[tuple[int, int]] = []
    for m in _DEFINITION.finditer(markdown):
        if protected(m.start("label")):
            continue
        label = _label(m.group("label"))
        definitions.setdefault(label, _destination(m.group("dest")))
        defined_spans.append((m.start(), m.end()))
        links.append(
            MarkdownLink(
                m.start(),
                m.end(),
                m.group("label"),
                _destination(m.group("dest")),
                "definition",
                label,
            )
        )
    in_definition = in_spans(defined_spans)
    for m in _LINK.finditer(markdown):
        if protected(m.start()) or in_definition(m.start()):
            continue
        if m.group("auto") is not None:
            links.append(
                MarkdownLink(
                    m.start(), m.end(), m.group("auto"), m.group("auto"), "autolink"
                )
            )
        elif m.group("short") is not None or m.group("label") is not None:
            text = m.group("short") if m.group("short") is not None else m["text"]
            label = _label(m.group("label") or text)
            if label in definitions:
                links.append(
                    MarkdownLink(
                        m.start(),
                        m.end(),
                        text,
                        definitions[label],
                        "reference",
                        label,
                    )
                )
        else:
            links.append(
                MarkdownLink(
                    m.start(),
                    m.end(),
                    m.group("text"),
                    _destination(m.group("dest") or ""),
                    "inline",
                )
            )
    return sorted(links, key=lambda link: link.start)


def link_key(url: str) -> str | None:
    """What two links to the same target share: a paper's key
    (:func:`paper_link_key`, so every arXiv or DOI form of one paper
    matches), else host (without ``www.``), path (without a trailing
    slash) and query, the scheme and fragment ignored. None for a target
    without a host."""
    url = url.strip()
    if key := paper_link_key(url):
        return key
    parsed = urlsplit(unquote(url))
    if not parsed.hostname:
        return None
    host = parsed.hostname.lower().removeprefix("www.")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{parsed.path.rstrip('/')}{query}"


# What may follow a URL a text states (not more of the URL).
_URL_END = re.compile(r"[^\w\-~%]|$")


@dataclass(frozen=True)
class LinkSources:
    """The targets a post may link: what its sources are or state. Build it
    with :meth:`of`; :meth:`allows` decides a target."""

    keys: frozenset[str] = frozenset()
    # Text of the sources (lower-cased): a non-arXiv URL it writes out is
    # stated (a paper names its repository or project page).
    stated: str = ""

    @classmethod
    def of(cls, urls: Iterable[str], *, stated_text: str = "") -> LinkSources:
        return cls(
            frozenset(k for u in urls if (k := link_key(u))), stated_text.lower()
        )

    def allows(self, target: str) -> bool:
        """Whether ``target`` (an absolute link) is a source or one the
        sources state. An arXiv link counts by its id only: a writer recalls
        ids from memory, and a source's text is no check of them."""
        key = link_key(target)
        if key is not None and key in self.keys:
            return True
        if key is None or key.startswith("arxiv:"):
            return False
        plain = unquote(target.strip()).rstrip("/").lower()
        start = self.stated.find(plain)
        while start != -1:
            if _URL_END.match(self.stated, start + len(plain)):
                return True
            start = self.stated.find(plain, start + 1)
        return False


def is_absolute_link(target: str) -> bool:
    """A link that leaves the site (not an anchor or a site path)."""
    return bool(_ABSOLUTE.match(target.strip()))


def verify_links(markdown: str, sources: LinkSources) -> tuple[str, list[str]]:
    """``markdown`` with every link (outside code and math) that leaves the
    site for a target ``sources`` does not allow reduced to its text, and
    the removed targets (each once, in order). A definition goes with the
    references to it; images are left to the figure checks. Writers link
    from memory: a recalled URL or arXiv id often names something else."""
    links = markdown_links(markdown)
    removed = list(
        dict.fromkeys(
            link.target
            for link in links
            if is_absolute_link(link.target) and not sources.allows(link.target)
        )
    )
    if not removed:
        return markdown, []
    gone = set(removed)
    out, pos = [], 0
    for link in links:
        if link.target not in gone:
            continue
        out.append(markdown[pos : link.start])
        if link.form != "definition":
            out.append(link.text)
        pos = link.end
    out.append(markdown[pos:])
    return "".join(out), removed


def unsourced_links(links: str, sources: LinkSources) -> tuple[str, list[str]]:
    """The links of ``links`` (a comma-separated list a writer copied "from
    the paper") that ``sources`` allow, joined again, and the targets of
    the others, which are dropped whole (a list item reduced to its text
    would be no link)."""
    kept: list[str] = []
    removed: list[str] = []
    for link in markdown_links(links):
        if sources.allows(link.target):
            kept.append(links[link.start : link.end])
        else:
            removed.append(link.target)
    return ", ".join(kept), removed
