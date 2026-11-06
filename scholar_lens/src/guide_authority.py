"""Deterministic source checks of the tech-guide pipeline: which flags,
arguments and metric names the official docs never mention.

No LLM or network calls: everything here reads text the pipeline already has.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..domain.markdown.math import segments
from .web_research import METRIC_SUFFIXES, OfficialDocs, ResearchCorpus

# Identifiers a reader copies verbatim: CLI flags and Prometheus-style metric
# names ("vllm:time_to_first_token_seconds", "..._requests_total").
_FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*")
_METRIC = re.compile(
    r"\b[a-z][a-z0-9]*(?::[a-z0-9]+(?:_[a-z0-9]+)+"
    rf"|(?:_[a-z0-9]+)+_(?:{'|'.join(METRIC_SUFFIXES)}))\b"
)
# Python API arguments of more than one word (``gpu_memory_utilization=``):
# the same settings as the flags, as a reader passes them to the product's API.
_KWARG = re.compile(r"(?<![\w.-])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)=(?!=)")
MAX_UNVERIFIED = 10


def _mentions(text: str, token: str) -> bool:
    """Whether ``text`` has ``token`` as a whole token (``--host`` is not in
    ``--hostname``)."""
    return re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", text) is not None


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


def _both_styles(known: str) -> bool:
    """Whether lowercased official text shows both "--a-b" and "--a_b" flags
    (torchrun, accelerate take both; ruff, which shows only dashes, rejects
    "_")."""
    return bool(re.search(r"--[a-z0-9]+-[a-z]", known)) and bool(
        re.search(r"--[a-z0-9]+_[a-z]", known)
    )


def _spellings(identifier: str, both_styles: bool) -> list[str]:
    """Ways the official text may spell a flag, a Python API argument
    (``a_b`` or its flag ``--a-b``) or a metric name (``a:b`` also as
    ``a_b``)."""
    if identifier.startswith("--"):
        return _flag_spellings(identifier, both_styles)
    if ":" in identifier:
        return [identifier, identifier.replace(":", "_")]
    return [identifier, "--" + identifier.replace("_", "-")]


def mentions_any(
    text: str, identifiers: Sequence[str], official_text: str = ""
) -> bool:
    """Whether ``text`` settles at least one of ``identifiers`` (flags, API
    arguments, metric names) the way :func:`unverified_identifiers` checks
    them, with the dash styles of ``official_text`` plus ``text``."""
    known = text.lower()
    both_styles = _both_styles(f"{official_text.lower()}\n{known}")
    return any(
        _mentions(known, spelling)
        for identifier in identifiers
        for spelling in _spellings(identifier.lower(), both_styles)
    )


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
    both_styles = _both_styles(known)
    found: list[str] = []

    def check(token: str) -> None:
        if token not in found and not any(
            _mentions(known, v) for v in _spellings(token, both_styles)
        ):
            found.append(token)

    for seg in segments(body):
        if seg.kind not in ("fence", "icode"):
            continue
        code = body[seg.start : seg.end]
        if seg.kind == "fence":
            # The opening fence's info string names a language ("```python"),
            # not a command run.
            code = code.partition("\n")[2]
        code = re.sub(r"\\\n\s*", " ", code)  # join lines
        names_product = any(pattern.search(code.lower()) for pattern in tokens)
        if (seg.kind == "fence" and names_product) or _KWARG.match(code.strip("` ")):
            for argument in _KWARG.findall(code):
                check(argument)
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
                check(flag)
            for metric in _METRIC.findall(line):
                check(metric)
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
