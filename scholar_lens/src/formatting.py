"""Assemble the published Markdown documents (Jekyll front matter + body).

Reviews, summaries and technical guides share one front-matter renderer; each
artifact adds its own body layout and reference list here, so the pipelines
only produce content and never format posts themselves.

Blog conventions encoded here: no ``author`` key (the blog's feed reads it as
the post author); paper posts name the publishing organization in
``paper_author`` and carry a translated ``subtitle``; the body never has an H1;
``use_math`` is set only when the post contains math.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import quote

from .markdown_math import has_math, normalize_math
from .post_conventions import PostConventions
from .tags import normalize_tags
from .utils import escape_yaml_double_quoted, is_placeholder
from .utils.helpers import first_sentence

if TYPE_CHECKING:
    from ..configs import Github as GithubConfig
    from .explainer import Paper, ReviewText
    from .tech_guide import TechGuide

_LEADING_H1 = re.compile(r"\A\s*#[ \t]+(?P<title>[^\n]+)\n?")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _quoted(value: str) -> str:
    return f'"{escape_yaml_double_quoted(value)}"'


def _quoted_list(values: list[str]) -> str:
    return ", ".join(_quoted(v) for v in values if v)


def _slug(value: str) -> str:
    """Category slug: entity-free, whitespace collapsed to hyphens."""
    return "-".join(html.unescape(value).split())


def render_front_matter(
    *,
    title: str,
    date: datetime,
    categories: list[str],
    tags: list[str],
    cover: str,
    use_math: bool,
    subtitle: str | None = None,
    paper_author: str | None = None,
    paper_doi: str | None = None,
    seo_title: str | None = None,
    description: str | None = None,
) -> str:
    """Render Jekyll front matter; every scalar is YAML-escaped and optional
    keys are omitted when empty."""
    lines = ["---", "layout: post", f"title: {_quoted(title)}"]
    if seo_title:
        lines.append(f"seo_title: {_quoted(seo_title)}")
    if subtitle:
        lines.append(f"subtitle: {_quoted(subtitle)}")
    lines.append(f"date: {date.strftime('%Y-%m-%d %H:%M:%S')}")
    if paper_author:
        lines.append(f"paper_author: {_quoted(paper_author)}")
    if paper_doi:
        lines.append(f"paper_doi: {_quoted(paper_doi)}")
    if description:
        lines.append(f"description: {_quoted(description)}")
    lines += [
        f"categories: [{_quoted_list(categories)}]",
        f"tags: [{_quoted_list(tags)}]",
        f"cover: {cover}",
    ]
    if use_math:
        lines.append("use_math: true")
    return "\n".join([*lines, "---", ""])


def split_leading_h1(body: str) -> tuple[str | None, str]:
    """Return ``(h1_text, rest)`` when the body opens with an H1 heading."""
    m = _LEADING_H1.match(body)
    if not m:
        return None, body
    return m.group("title").strip(), body[m.end() :]


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


def _paper_author(paper: Paper) -> str | None:
    """The organization that published the paper (never people's names)."""
    affiliation = html.unescape(paper.attributes.affiliation).strip()
    return None if is_placeholder(affiliation) else affiliation


@dataclass
class PostMeta:
    """Taxonomy decided in code from the model's proposals and the spec."""

    topic: str | None
    cover: str
    tags: list[str]
    seo_title: str | None = None
    new_tags: list[str] = field(default_factory=list)


def resolve_topic(raw: str, spec: PostConventions) -> str | None:
    """Map a proposed category onto a spec topic (exact, slug or case match)."""
    wanted = _slug(raw).lower()
    for topic in spec.topics:
        if topic.lower() == wanted:
            return topic
    return None


def cut_title(title: str, limit: int) -> str:
    """``title`` (whitespace collapsed) cut to ``limit`` at a word boundary."""
    title = " ".join(title.split())
    if len(title) <= limit:
        return title
    cut = title[: limit + 1].rsplit(" ", 1)[0].rstrip(" :,-–—")
    return cut or title[:limit]


def shorten_title(title: str, limit: int) -> str | None:
    """A paper's ``seo_title`` for a title over ``limit``: the part before
    the colon (the paper's name) when it is usable, else the title cut at a
    word boundary (:func:`cut_title`)."""
    title = " ".join(title.split())
    if len(title) <= limit:
        return None
    head = title.split(":", 1)[0].strip()
    if ":" in title and 3 <= len(head) <= limit:
        return head
    return cut_title(title, limit)


def build_post_meta(
    *,
    title: str,
    category_type: str,
    proposed_topic: str,
    specific_tags: list[str],
    controlled_tags: list[str],
    proposed_seo_title: str | None,
    spec: PostConventions,
    known_tags: list[str] | None = None,
) -> PostMeta:
    """Normalize tags, resolve the topic/cover and pick a ``seo_title``.

    Anything that still breaks a convention (no controlled tag, unknown topic)
    is left for the post validator to reject, never silently patched.
    """
    topic = resolve_topic(proposed_topic, spec)
    selection = normalize_tags(
        specific_tags, controlled_tags, title=title, spec=spec, known_tags=known_tags
    )
    limit = spec.seo_title_max_length
    seo = (proposed_seo_title or "").strip() or None
    if len(title) <= limit:
        seo = None
    elif not seo or len(seo) > limit:
        seo = shorten_title(title, limit)
    new_tags = (
        [t for t in selection.specific if t not in set(known_tags)]
        if known_tags is not None
        else []
    )
    return PostMeta(
        topic=topic,
        cover=spec.cover_for(category_type, topic or ""),
        tags=selection.tags,
        seo_title=seo,
        new_tags=new_tags,
    )


def paper_post_meta(
    paper: Paper,
    category_type: str,
    spec: PostConventions,
    known_tags: list[str] | None = None,
) -> PostMeta:
    attributes = paper.attributes
    return build_post_meta(
        title=paper.title,
        category_type=category_type,
        proposed_topic=html.unescape(attributes.category),
        specific_tags=attributes.keywords,
        controlled_tags=attributes.controlled_tags,
        proposed_seo_title=attributes.seo_title,
        spec=spec,
        known_tags=known_tags,
    )


def paper_front_matter(
    paper: Paper,
    category_type: str,
    meta: PostMeta,
    *,
    body: str = "",
    subtitle: str | None = None,
    description: str | None = None,
) -> str:
    """Front matter for a review or summary of ``paper``; ``body`` decides
    ``use_math``."""
    categories = [category_type] + ([meta.topic] if meta.topic else [])
    return render_front_matter(
        title=paper.title,
        seo_title=meta.seo_title,
        subtitle=subtitle,
        date=paper.published,
        paper_author=_paper_author(paper),
        # The blog looks up the paper's live citation count by it.
        paper_doi=paper.doi,
        description=description,
        categories=categories,
        tags=meta.tags,
        cover=meta.cover,
        use_math=has_math(body),
    )


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


def _paper_references(paper: Paper, extra_links: list[tuple[str, str]]) -> str:
    links = [
        (paper.title, str(paper.pdf_url)),
        *(
            ("/".join(str(r).rstrip("/").split("/")[-2:]) or str(r), str(r))
            for r in paper.repo_urls
        ),
        *extra_links,
    ]
    return f"\n\n- - -\n### References\n{link_list(links)}\n"


def same_as_title(candidate: str, title: str) -> bool:
    """A subtitle that only repeats the title (case and spacing ignored)."""
    return " ".join(candidate.split()).casefold() == " ".join(title.split()).casefold()


# A derived subtitle keeps a whole first sentence up to _SUBTITLE_MAX
# characters; longer ones are cut near _SUBTITLE_CUT (clause or word boundary)
# and end with "…".
_SUBTITLE_MAX = 80
_SUBTITLE_CUT = 60
_QUESTION_LABEL = re.compile(r"^\*\*[^*\n]*\?\*\*\s*")
_LABEL = re.compile(r"^\*\*[^*\n]{1,30}:\*\*\s*|^\*\*[^*\n]{1,30}\*\*:\s*")


def _plain(text: str) -> str:
    """Inline Markdown reduced to plain text: links keep their text, code
    keeps its content, paired emphasis markers go (an intraword ``_`` stays)."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`+([^`]*)`+", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    return " ".join(text.split())


def _shorten_subtitle(text: str) -> str:
    if len(text) <= _SUBTITLE_MAX:
        return text
    head = text[: _SUBTITLE_MAX + 1]
    for boundary in (", ", "; ", " "):
        cut = head.rfind(boundary, _SUBTITLE_CUT // 2, _SUBTITLE_MAX)
        if cut > 0:
            return head[:cut].rstrip(" ,;:") + "…"
    return head[:_SUBTITLE_CUT].rstrip(" ,;:") + "…"


def _lead_sentence(markdown: str) -> str:
    """The first prose sentence of ``markdown`` as plain text (headings,
    tables, math and labels skipped): a last-resort subtitle."""
    for line in markdown.splitlines():
        text = line.strip()
        if not text or text.startswith(("#", "|", "!", "```", "~~~", "- - -")):
            continue
        if "$" in text:
            continue  # a subtitle is plain text; math would leave holes
        text = re.sub(r"^(?:>\s*|[-*]\s+|\d+[.)]\s+)", "", text)
        text = _QUESTION_LABEL.sub("", text)
        text = _LABEL.sub("", text)
        text = first_sentence(_plain(text)).rstrip(".!?。 ")
        if text:
            return _shorten_subtitle(text)
    return ""


def derived_subtitle(explicit: str | None, h1: str | None, title: str) -> bool:
    """True when neither the model's subtitle nor the body's H1 is usable, so
    the post falls back to its first sentence (worth a reviewer's look)."""
    return not any(
        c and c.strip() and not same_as_title(c, title) for c in (explicit, h1)
    )


def _subtitle(
    explicit: str | None, h1: str | None, title: str, fallback: str = ""
) -> str | None:
    """The subtitle: the model's, else the body's H1, else the first sentence
    of ``fallback`` (the TL;DR); never one that just repeats the title."""
    for candidate in (explicit, h1, _lead_sentence(fallback)):
        if candidate and candidate.strip() and not same_as_title(candidate, title):
            return candidate.strip()
    return None


def format_review(
    github_config: GithubConfig,
    paper: Paper,
    review: ReviewText,
    *,
    meta: PostMeta,
) -> str:
    """Front matter → ``## TL;DR`` → body → closing assessment → references."""
    h1, body = split_leading_h1(review.body)
    parts = [f"## TL;DR\n{review.key_takeaways.strip()}", "- - -", body.strip()]
    if review.closing.strip():
        parts.append(review.closing.strip())
    content = normalize_math("\n\n".join(parts))
    front_matter = paper_front_matter(
        paper,
        github_config.review_category,
        meta,
        body=content,
        subtitle=_subtitle(
            review.subtitle, h1, paper.title, fallback=review.key_takeaways
        ),
        description=review.description,
    )
    return f"{front_matter}{content}{_paper_references(paper, [])}"


_LEAD_BLOCKQUOTE = re.compile(r"\A((?:>[^\n]*\n?)+)")
_SUMMARY_LABEL = re.compile(r"^\*\*[^*]{1,20}:?\*\*:?\s*")


def _summary_lead(body: str) -> tuple[str, str]:
    """``(lead, rest)``: the text of the summary's leading ``> **요약:** …``
    blockquote (label dropped) and the body after it."""
    match = _LEAD_BLOCKQUOTE.match(body)
    if not match:
        return "", body
    lines = [line[1:].strip() for line in match.group(1).splitlines()]
    text = _SUMMARY_LABEL.sub("", " ".join(line for line in lines if line))
    return text, body[match.end() :].lstrip("\n")


def summary_lead(body: str) -> str:
    """The summary's TL;DR text (its leading blockquote)."""
    return _summary_lead(body.lstrip())[0]


def summary_with_tldr(body: str) -> str:
    """Paper posts open with ``## TL;DR``: turn the summary's leading
    ``> **요약:** …`` blockquote into that section (label dropped)."""
    if body.startswith("## TL;DR"):
        return body
    text, rest = _summary_lead(body)
    return f"## TL;DR\n\n{text}\n\n{rest}" if text else rest


def format_summary(
    github_config: GithubConfig,
    paper: Paper,
    result: dict[str, str],
    *,
    meta: PostMeta,
) -> str:
    """Front matter → ``## TL;DR`` → summary sections → references."""
    h1, body = split_leading_h1(result["summary"])
    content = normalize_math(summary_with_tldr(body.strip()))
    front_matter = paper_front_matter(
        paper,
        github_config.summary_category,
        meta,
        body=content,
        subtitle=_subtitle(result.get("subtitle"), h1, paper.title, fallback=content),
        description=result.get("description"),
    )
    links = _MD_LINK.findall(result.get("urls", ""))
    return f"{front_matter}{content}{_paper_references(paper, links)}"


def format_guide(
    github_config: GithubConfig,
    guide: TechGuide,
    *,
    meta: PostMeta,
) -> str:
    """Front matter (``["Tech Guides", <topic>]``) → guide body → sources,
    each linked by its page title (``guide.source_titles``)."""
    _, body = split_leading_h1(guide.body)
    content = normalize_math(body.strip())
    categories = [github_config.tech_guide_category] + (
        [meta.topic] if meta.topic else []
    )
    front_matter = render_front_matter(
        title=guide.topic,
        seo_title=meta.seo_title,
        date=datetime.now(UTC),
        categories=categories,
        tags=meta.tags,
        cover=meta.cover,
        use_math=has_math(content),
    )
    titles = guide.source_titles
    sources = link_list((titles.get(url), url) for url in guide.source_urls)
    return f"{front_matter}{content}\n\n- - -\n### Sources\n{sources}\n"
