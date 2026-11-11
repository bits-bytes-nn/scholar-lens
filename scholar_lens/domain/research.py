"""What web research gathers for a guide or survey: pages and search results,
and the excerpts of them the prompts read (pure data and text rules)."""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections.abc import Collection
from dataclasses import dataclass, field
from itertools import pairwise

from scholar_lens.domain.ids import arxiv_id_from_url, arxiv_paper_key
from scholar_lens.domain.markdown.lint import close_open_fence
from scholar_lens.domain.markdown.math import segments
from scholar_lens.domain.prompt_safety import neutralize_prompt_tags
from scholar_lens.domain.sources import normalize_url
from scholar_lens.domain.text import parse_lines

# How a source is labelled in a prompt. Official pages decide commands, flags
# and defaults; reference material (chosen by the user) and third-party pages
# ground concepts, internals and architecture.
OFFICIAL_LABEL = "official documentation"


OLDER_DOCS_LABEL = "older docs version"


REFERENCE_LABEL = "reference material"


THIRD_PARTY_LABEL = "third-party"


# A page joins a prompt's sources only with more room than this left.
_MIN_SOURCE_ROOM = 2_000


def arxiv_paper_urls(url: str) -> tuple[str, str] | None:
    """For a link to an arXiv paper (abs, pdf or html, on arXiv or a mirror),
    the paper's HTML rendering (the full text; the abstract page holds only
    the abstract and a PDF is not read) and its abstract page, the fallback;
    None for any other URL."""
    arxiv_id = arxiv_id_from_url(url)
    if arxiv_id is None:
        return None
    return f"https://arxiv.org/html/{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}"


def page_key(url: str) -> str:
    """Dedup key of a page: one per arXiv paper (its abs, pdf and html links
    and every version), else :func:`normalize_url`."""
    paper = arxiv_paper_key(url)
    return f"arxiv:{paper}" if paper else normalize_url(url)


def product_names(proposed: str, pages: Collection[PageContent]) -> list[str]:
    """The command and package names the relevance gate read off the docs
    (``proposed``, comma- or line-separated), kept only when the seed
    ``pages`` show them as a word. No name shown means none: a product name
    is never guessed from the site's host."""
    text = " ".join(page.text.lower() for page in pages)
    names: list[str] = []
    # A "none" answer names nothing (any Python page shows "None").
    for raw in parse_lines("\n".join(re.split(r"[,\n]", proposed))):
        name = raw.strip("`'\"").lower()
        if (
            name
            and name not in names
            and re.search(rf"(?<![\w/.-]){re.escape(name)}(?![\w-])", text)
        ):
            names.append(name)
    return names


class SourceFetchError(Exception):
    """Every requested page failed for transport reasons (timeouts, refused
    connections, 429/5xx): a retryable outage, not a bad request."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    description: str = ""
    # Cleaned, extracted page body when the search backend provides it (Tavily).
    # Lets the corpus use the result directly instead of re-fetching the URL.
    content: str = ""
    # When the backend dates the page (Brave's ``page_age``, Tavily's
    # ``published_date``): ``YYYY-MM-DD``, else "".
    published: str = ""


@dataclass
class PageContent:
    url: str
    title: str
    text: str
    image_urls: list[str] = field(default_factory=list)
    # Each image's caption (its figure's caption, else its alt text; ""
    # for arXiv's "Refer to caption" placeholder).
    image_captions: dict[str, str] = field(default_factory=dict)
    links: list[str] = field(default_factory=list)
    # The links inside the page's single <article>/<main> (all of ``links``
    # when it has none): what its text cites, without its site navigation.
    content_links: list[str] = field(default_factory=list)
    # Where the page actually lives after HTTP and <meta refresh> redirects
    # (sub-page scope is taken from it); empty when it is ``url`` itself.
    final_url: str = ""
    # Explanatory material the user passed with the request (a paper, design
    # doc or deep-dive post): labelled reference material, never official.
    user_reference: bool = False
    # The source record (:mod:`source_set`), decided once at gather time by
    # :meth:`SourceSet.record`; "" until then. ``published`` is the page's
    # own date (its metadata, else the search backend's), ``YYYY-MM-DD``.
    authority: str = ""
    version: str = ""
    published: str = ""
    language: str = ""


@dataclass
class ResearchCorpus:
    """All gathered material for a guide."""

    pages: list[PageContent] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)

    @property
    def image_urls(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for page in self.pages:
            for url in page.image_urls:
                if url not in seen:
                    seen.add(url)
                    ordered.append(url)
        return ordered

    def relevant_text(
        self,
        query: str,
        *,
        max_chars: int,
        labelled: bool = False,
        prefer: Collection[str] = (),
    ) -> str:
        """The pages most relevant to ``query`` (word overlap with title and
        body), most relevant first, within ``max_chars``; a page over its room
        keeps its most relevant parts (:func:`_excerpt`). The pages whose URL
        is in ``prefer`` (the section plan's) come first, in that order, each
        with an equal share of ``max_chars`` so a long paper cannot push an
        official page out; the room they leave goes to the other pages by
        rank, a cut preferred page first. Each section of a guide reads only
        the material it needs instead of the whole corpus.
        With ``labelled``, each source is labelled by its record (official,
        another docs version, reference material or third-party) so the
        writer knows which one decides a flag or default."""
        words = {w for w in re.findall(r"[a-z0-9][a-z0-9_.-]{2,}", query.lower())}

        def score(page: PageContent) -> float:
            title = page.title.lower()
            body = page.text.lower()
            hits = sum(3 * (w in title) + min(body.count(w), 20) for w in words)
            return hits / (1 + len(body) / 20_000)  # don't favour sheer length

        by_url = {page.url: page for page in reversed(self.pages)}
        preferred = [by_url[url] for url in dict.fromkeys(prefer) if url in by_url]
        chosen = {id(page) for page in preferred}
        others = sorted(
            (p for p in self.pages if id(p) not in chosen), key=score, reverse=True
        )
        ordered = preferred + others
        blocks = [_source_block(page, labelled) for page in ordered]
        share = max_chars // len(preferred) if preferred else 0
        cuts = [_excerpt(blocks[i], share, words) for i in range(len(preferred))]
        cuts += [""] * len(others)
        used = sum(map(len, cuts))
        for i, block in enumerate(blocks):
            if max_chars - used <= _MIN_SOURCE_ROOM:
                break
            if len(cuts[i]) < len(block):
                longer = _excerpt(block, max_chars - used + len(cuts[i]), words)
                if len(longer) > len(cuts[i]):
                    used += len(longer) - len(cuts[i])
                    cuts[i] = longer
        return "\n\n---\n\n".join(cut for cut in cuts if cut)


_HEADING_LINE = re.compile(r"^#{1,6}[ \t]", re.MULTILINE)


_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


_ELISION = "\n\n…\n\n"


def _parts(text: str, limit: int) -> list[tuple[str, int | None]]:
    """``text`` split at its heading lines, a part still over ``limit`` at its
    paragraph breaks; never inside a code fence or a ``$$`` span. Each part
    comes with the index of its heading section's first part (the one with
    the heading line), ``None`` before the first heading."""
    protected = [
        (seg.start, seg.end)
        for seg in segments(text)
        if seg.kind in ("fence", "display")
    ]
    starts = [start for start, _ in protected]

    def free(pos: int) -> bool:
        i = bisect_right(starts, pos) - 1
        return i < 0 or pos >= protected[i][1] or pos == protected[i][0]

    def spans(start: int, end: int, cuts: list[int]) -> list[tuple[int, int]]:
        bounds = [start, *(c for c in cuts if start < c < end), end]
        return list(pairwise(bounds))

    headings = [m.start() for m in _HEADING_LINE.finditer(text) if free(m.start())]
    breaks = [m.end() for m in _PARAGRAPH_BREAK.finditer(text) if free(m.end())]
    heading_starts = set(headings)
    parts: list[tuple[str, int | None]] = []
    for start, end in spans(0, len(text), headings):
        pieces = spans(start, end, breaks) if end - start > limit else [(start, end)]
        first = len(parts) if start in heading_starts else None
        parts += [(text[a:b], first) for a, b in pieces if text[a:b].strip()]
    return parts


def _excerpt(block: str, limit: int, words: Collection[str]) -> str:
    """A source block within ``limit``: whole when it fits, else its header
    (the source and title lines) and the parts of its text (:func:`_parts`)
    that hold the most of ``words``, each weighted by how few parts hold it
    (inverse document frequency), in page order, each gap marked by a "…"
    line; a kept piece of a long section brings the section's heading piece
    ("Deprecated options (removed in v0.10)") along. A reference page lists
    the flags a section needs anywhere on it, so keeping only its start
    would drop them."""
    if len(block) <= limit:
        return block
    header, sep, text = block.partition("\n\n")
    # Room for the parts, each with a join and a "…" line, and a last "…".
    room = limit - len(header) - len(sep) - len(_ELISION)
    if not sep or room <= len(_ELISION):
        return _cut(block, limit)
    pieces = _parts(text, room)
    parts = [part for part, _ in pieces]
    lowered = [part.lower() for part in parts]
    # A word in every part (the product's name, "and") picks none of them; a
    # word in few parts (a flag's) picks those.
    weight = {
        w: math.log(len(parts) / df)
        for w in words
        if (df := sum(w in part for part in lowered))
    }

    def score(i: int) -> float:
        return sum(v for w, v in weight.items() if w in lowered[i])

    ranked = sorted(range(len(parts)), key=lambda i: (-score(i), i))
    kept: dict[int, str] = {}
    for i in ranked:
        if i in kept:
            continue
        first = pieces[i][1]
        wanted = [j for j in (first, i) if j is not None and j not in kept]
        cost = sum(len(parts[j]) + len(_ELISION) for j in set(wanted))
        if cost <= room:
            kept.update((j, parts[j]) for j in wanted)
            room -= cost
        elif not kept:  # the best part alone is too long: its start
            kept[i] = _cut(parts[i], room - len(_ELISION))
            room = 0
    out: list[str] = []
    previous = -1
    for i in sorted(kept):
        if i != previous + 1:
            out.append(_ELISION.strip("\n"))
        out.append(kept[i].strip("\n"))
        previous = i
    if previous != len(parts) - 1:
        out.append(_ELISION.strip("\n"))
    return header + sep + "\n\n".join(part for part in out if part)


def _source_block(page: PageContent, labelled: bool) -> str:
    """One page for a prompt: URL, title and text, defanged (all three are
    page-controlled), labelled by :func:`source_label` when ``labelled``."""
    label = f" ({source_label(page)})" if labelled else ""
    return (
        f"## Source{label}: {neutralize_prompt_tags(page.url)}\n"
        f"# {neutralize_prompt_tags(page.title)}\n\n"
        f"{neutralize_prompt_tags(page.text)}"
    )


# How a prompt names each record authority.
_LABELS: dict[str, str] = {
    "official": OFFICIAL_LABEL,
    "official_repo": OFFICIAL_LABEL,
    "reference": REFERENCE_LABEL,
    "third_party": THIRD_PARTY_LABEL,
    "forum": THIRD_PARTY_LABEL,
    "older_docs": OLDER_DOCS_LABEL,
    "dev_docs": OLDER_DOCS_LABEL,
}


def source_label(page: PageContent) -> str:
    """How a prompt labels ``page``, read off its record (third-party until
    the page is recorded): its docs or repository are official, another docs
    version is an older docs version."""
    return _LABELS.get(page.authority, THIRD_PARTY_LABEL)


def _cut(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` at a paragraph break, never inside a code
    fence (a fence left open would make the rest of the prompt read as code)."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    if (paragraph := head.rfind("\n\n")) > limit // 2:
        head = head[:paragraph]
    # Cut inside a math span: drop the open one. Code is masked first, since a
    # shell ``$$`` (the PID) is not math and would otherwise cut the page.
    masked = list(head)
    for seg in segments(head):
        if seg.kind in ("fence", "icode"):
            masked[seg.start : seg.end] = " " * (seg.end - seg.start)
    prose = "".join(masked)
    if prose.count("$$") % 2:
        head = head[: prose.rfind("$$")].rstrip()
        prose = prose[: len(head)]
    # A cut inside inline code leaves an odd backtick, inside "$$" a lone "$".
    if prose.count("`") % 2:
        head = head[: prose.rfind("`")].rstrip()
        prose = prose[: len(head)]
    # Only a "$$" split by the cut leaves a lone "$"; "$x$" ends correctly.
    if re.search(r"(?<!\$)\$$", prose) and text[len(head) : len(head) + 1] == "$":
        head = head[:-1].rstrip()
    return close_open_fence(head)


# Prometheus naming-convention suffixes that mark a metric name
# ("..._requests_total"); guide_authority's metric pattern uses the same list.
METRIC_SUFFIXES = ("total", "seconds", "bytes", "count", "bucket", "sum")
