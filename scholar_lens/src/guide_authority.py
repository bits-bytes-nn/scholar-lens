"""Deterministic source checks of the tech-guide pipeline: which flags,
arguments and metric names the official docs never mention, which pages a
guide draws on, and the synopsis's per-section source and concept plan.

No LLM or network calls: everything here reads text the pipeline already has.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from .markdown_math import segments
from .prompt_safety import neutralize_prompt_tags
from .review_plan import parse_lines
from .web_research import (
    OfficialDocs,
    PageContent,
    ResearchCorpus,
    docs_version,
    normalize_url,
)

# Identifiers a reader copies verbatim: CLI flags and Prometheus-style metric
# names ("vllm:time_to_first_token_seconds", "..._requests_total").
_FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*")
_METRIC = re.compile(
    r"\b[a-z][a-z0-9]*(?::[a-z0-9]+(?:_[a-z0-9]+)+"
    r"|(?:_[a-z0-9]+)+_(?:total|seconds|bytes|count|bucket|sum))\b"
)
# Python API arguments of more than one word (``gpu_memory_utilization=``):
# the same settings as the flags, as a reader passes them to the product's API.
_KWARG = re.compile(r"(?<![\w.-])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)=(?!=)")
MAX_UNVERIFIED = 10


_OUTLINE_HEADING = re.compile(r"^##[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_MAX_OUTLINE_HEADINGS = 40
# "1 | sources: …", also "1. |", "3|" or "Section 4 |".
_PLAN_LINE = re.compile(r"^\s*(?:section\s+)?(\d+)[.)]?(?:\s+|(?=\|))(.*)$", re.I)
# A URL, bare, in <…> or in a Markdown link "[title](url)".
_PLAN_URL = re.compile(r"https?://[^\s,;|<>\[\]]+")
_URL_TRAILER = ").,;:!?'\"`*"


def _mentions(text: str, token: str) -> bool:
    """Whether ``text`` has ``token`` as a whole token (``--host`` is not in
    ``--hostname``)."""
    return re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", text) is not None


def input_outline(corpus: ResearchCorpus, urls: tuple[str, ...] | list[str]) -> str:
    """The ``##`` headings of the requested pages (not sub-pages or search
    results), one per line: the topics a getting-started guide must cover."""
    wanted = {normalize_url(u) for u in urls}
    headings: list[str] = []
    for page in corpus.pages:
        if normalize_url(page.url) not in wanted:
            continue
        for title in _OUTLINE_HEADING.findall(page.text):
            if title not in headings:
                headings.append(title)
    lines = [f"- {neutralize_prompt_tags(h)}" for h in headings[:_MAX_OUTLINE_HEADINGS]]
    return "\n".join(lines) or "(none)"


def parse_section_plan(
    raw: str, pages: Collection[PageContent]
) -> dict[int, tuple[list[str], list[str]]]:
    """``{section number: (source URLs, owned concepts)}`` from the synopsis's
    ``<section_plan>`` lines (``1 | sources: url, url | owns: a; b``),
    numbered like the outline's sections. A URL
    counts only when it is a corpus page (as fetched or where it landed),
    returned as that page's URL; unknown URLs are dropped."""
    known: dict[str, str] = {}
    for page in pages:
        for url in (page.url, page.final_url):
            if url:
                known.setdefault(normalize_url(url), page.url)
    plan: dict[int, tuple[list[str], list[str]]] = {}
    for line in raw.splitlines():
        match = _PLAN_LINE.match(line)
        if not match:
            continue
        sources: list[str] = []
        owns: list[str] = []
        for part in match.group(2).split("|"):
            key, _, value = part.partition(":")
            key = key.strip().lower()
            if key == "sources":
                for url in _PLAN_URL.findall(value):
                    page_url = known.get(normalize_url(url.rstrip(_URL_TRAILER)))
                    if page_url and page_url not in sources:
                        sources.append(page_url)
            elif key == "owns":
                owns += parse_lines("\n".join(value.split(";")))
        plan[int(match.group(1))] = (sources, owns)
    return plan


def official_reference(official: OfficialDocs, corpus: ResearchCorpus) -> str:
    """Text of the fetched pages of the requested docs (where they landed, in
    a compatible docs version)."""
    return "\n".join(
        p.text for p in corpus.pages if official.covers(p.final_url or p.url)
    )


def _flag_spellings(flag: str, both_styles: bool) -> list[str]:
    """Ways the official text may spell ``flag``: as written, as a Python API
    argument (``--max-model-len`` → ``max_model_len``, multi-word names only,
    since a bare word like ``model`` appears in any prose), and in the other
    dash style when the docs use both."""
    name = flag[2:]
    spellings = [flag]
    if "-" in name:  # an underscore flag already is its own API form
        spellings.append(name.replace("-", "_"))
    if re.search(r"[-_]", name) and both_styles:
        spellings += ["--" + name.replace("_", "-"), "--" + name.replace("-", "_")]
    return spellings


def unverified_identifiers(
    body: str, official_text: str, products: Sequence[str]
) -> list[str]:
    """CLI flags of ``products`` commands, their Python API arguments and
    metric names the guide shows in code that the official pages never
    mention (a stale or invented flag is the costliest guide error: readers
    paste it). Flags count only after a product's name on a command line (or
    alone in inline code), so ``docker``/``curl`` options are not checked;
    arguments only in a code block that names the product (``from vllm import
    LLM``) or alone in inline code. Note-only: an empty ``official_text``
    checks nothing.
    """
    if not official_text.strip():
        return []
    names = [n.lower() for n in products if n]
    tokens = [re.compile(rf"(?<![\w/.-]){re.escape(n)}(?![\w-])") for n in names]
    known = official_text.lower()
    # CLIs that take both "--a-b" and "--a_b" (torchrun, accelerate) show both
    # styles in their docs; one that shows only dashes (ruff) rejects "_".
    both_styles = bool(re.search(r"--[a-z0-9]+-[a-z]", known)) and bool(
        re.search(r"--[a-z0-9]+_[a-z]", known)
    )
    found: list[str] = []

    def check(token: str, variants: list[str]) -> None:
        if token not in found and not any(_mentions(known, v) for v in variants):
            found.append(token)

    for seg in segments(body):
        if seg.kind not in ("fence", "icode"):
            continue
        code = re.sub(r"\\\n\s*", " ", body[seg.start : seg.end])  # join lines
        names_product = any(pattern.search(code.lower()) for pattern in tokens)
        if (seg.kind == "fence" and names_product) or _KWARG.match(code.strip("` ")):
            for argument in _KWARG.findall(code):
                check(argument, [argument, "--" + argument.replace("_", "-")])
        for line in code.splitlines():
            lowered = line.lower()
            # The product as a command word, not inside HF_TOKEN or a path.
            hits = [
                m.start() for m in (pattern.search(lowered) for pattern in tokens) if m
            ]
            start = min(hits) if hits else (0 if not names else -1)
            if seg.kind == "fence" and start < 0:
                continue
            # Inline code without the product counts a flag only when it is a
            # bare flag (`--max-model-len`), not another tool's (`curl -s`).
            bare_flag = line.strip("` ").startswith("--")
            flags = (
                _FLAG.findall(line[max(start, 0) :]) if start >= 0 or bare_flag else []
            )
            for flag in flags:
                check(flag, _flag_spellings(flag, both_styles))
            for metric in _METRIC.findall(line):
                check(metric, [metric, metric.replace(":", "_")])
    return found[:MAX_UNVERIFIED]


def third_party_identifiers(
    corpus: ResearchCorpus, official: OfficialDocs, products: Sequence[str]
) -> list[str]:
    """Flags and API arguments the corpus's other pages (blogs, older docs
    versions) show for the product that the official pages never mention:
    what the official reference must settle before a section is written."""
    official_text = official_reference(official, corpus)
    found: list[str] = []
    for page in corpus.pages:
        if official.covers(page.final_url or page.url):
            continue
        for identifier in unverified_identifiers(page.text, official_text, products):
            if identifier not in found:
                found.append(identifier)
    return found[:MAX_UNVERIFIED]


def used_source_urls(
    body: str,
    corpus: ResearchCorpus,
    urls: list[str] | tuple[str, ...],
    official: OfficialDocs,
    reference_pages: list[PageContent],
    section_sources: Collection[str],
) -> list[str]:
    """The pages the guide draws on, for its source list: the requested pages
    (``urls``: documentation and reference material) and the official pages
    crawled with them, and any other page that fed a
    section's sources (``section_sources``) or that the body links or shows
    an image from. A reference page counts only for an identifier it settled;
    an older docs version never counts."""
    requested = {normalize_url(u) for u in urls}
    references = {normalize_url(p.url) for p in reference_pages}
    code = " ".join(
        body[seg.start : seg.end]
        for seg in segments(body)
        if seg.kind in ("fence", "icode")
    )
    # Each flag or argument in both spellings (``--a-b`` / ``a_b``; a one-word
    # flag only as itself, since "host" is in any page).
    identifiers = {
        spelling
        for flag in _FLAG.findall(code)
        for spelling in _flag_spellings(flag, False)
    }
    identifiers |= {
        a for arg in _KWARG.findall(code) for a in (arg, "--" + arg.replace("_", "-"))
    }
    linked = {normalize_url(u) for u in re.findall(r"\]\((https?://[^)\s]+)", body)}
    used: list[str] = []
    seen: set[str] = set()
    for page in corpus.pages:
        norm = normalize_url(page.url)
        where = page.final_url or page.url
        text = page.text.lower()
        if norm in requested:
            keep = True
        elif official.on_host(where) and not official.covers(where):
            keep = False  # an older docs version
        elif norm in references:
            keep = any(_mentions(text, i) for i in identifiers)
        elif official.covers(where):
            keep = True  # a current page of the requested docs
        else:
            keep = (
                page.url in section_sources
                or norm in linked
                or any(u in body for u in page.image_urls)
            )
        if keep and _same_page_key(page.url) not in seen:
            seen.add(_same_page_key(page.url))
            used.append(page.url)
    return used


def _same_page_key(url: str) -> str:
    """One key for the spellings of one docs page: with or without ``.html``,
    under ``latest`` or ``stable``."""
    key = normalize_url(url).removesuffix(".html")
    version = docs_version(url)
    return key.replace(f"/{version}/", "/{v}/", 1) if version else key
