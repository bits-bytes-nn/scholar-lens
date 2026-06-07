r"""Lint + auto-fix a published Markdown post for the blog's kramdown/MathJax quirks.

A companion to :mod:`markdown_math`. Where that module escapes underscores, this
one catches the *other* recurring blog-rendering failures. It reuses the same
validated code/math-span splitter (:data:`markdown_math.SEGMENT_PATTERN`) so it
never mistakes code for prose or math.

Two classes of action, split by risk:

* **Auto-fixes** (deterministic, lossless) — applied silently:
    1. Insert the blank line kramdown needs before a heading glued to the
       previous line (outside fenced code).
    2. Replace a bare ``|`` inside a math span with ``\vert`` (kramdown otherwise
       reads it as a table delimiter); ``\vert`` renders identically.

* **Warnings** (context-dependent, NOT auto-changed — a wrong rewrite would ship
  a new bug) — logged so a human can review:
    3. A single-``$ ... $`` inline math span (the blog strips ``$`` delimiters,
       so it won't render — but it could also be a currency amount, so we don't
       touch it).
    4. A LaTeX macro outside the blog's known set (renders as red raw text).
    5. A Markdown link whose target is not an http(s) URL (a prose/section/slug
       target is a broken link).

:func:`lint_markdown` returns the auto-fixed text and logs any warnings.
"""

from __future__ import annotations

import re

from .logger import logger
from .markdown_math import SEGMENT_PATTERN

# Macros MathJax provides out of the box are too many to list; instead we only
# flag macros NOT known to render on this blog. The blog's custom/loaded set
# (beyond base MathJax) is small and explicit — extend here if the blog adds
# more \newcommand/packages.
_BLOG_EXTRA_MACROS = frozenset({"llbracket", "rrbracket", "textsc", "vert", "mid"})
# Common base-MathJax macros we never want to warn about. This is a pragmatic
# allowlist of frequently-used commands; anything outside it AND outside the
# blog set is surfaced as a warning (not an error), so a missing entry only
# costs a noisy log line, never a wrong rewrite.
_BASE_MATHJAX_MACROS = frozenset(
    {
        # greek
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "varepsilon",
        "zeta",
        "eta",
        "theta",
        "vartheta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "pi",
        "varpi",
        "rho",
        "varrho",
        "sigma",
        "varsigma",
        "tau",
        "upsilon",
        "phi",
        "varphi",
        "chi",
        "psi",
        "omega",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Xi",
        "Pi",
        "Sigma",
        "Upsilon",
        "Phi",
        "Psi",
        "Omega",
        # operators / functions
        "sum",
        "prod",
        "int",
        "iint",
        "iiint",
        "oint",
        "lim",
        "inf",
        "sup",
        "max",
        "min",
        "arg",
        "argmax",
        "argmin",
        "log",
        "ln",
        "exp",
        "sin",
        "cos",
        "tan",
        "det",
        "dim",
        "ker",
        "deg",
        "gcd",
        "Pr",
        # symbols / relations
        "infty",
        "partial",
        "nabla",
        "cdot",
        "cdots",
        "ldots",
        "dots",
        "vdots",
        "ddots",
        "times",
        "div",
        "pm",
        "mp",
        "ast",
        "star",
        "circ",
        "bullet",
        "oplus",
        "otimes",
        "odot",
        "leq",
        "geq",
        "neq",
        "approx",
        "equiv",
        "sim",
        "simeq",
        "cong",
        "propto",
        "ll",
        "gg",
        "subset",
        "supset",
        "subseteq",
        "supseteq",
        "in",
        "ni",
        "notin",
        "cup",
        "cap",
        "emptyset",
        "forall",
        "exists",
        "neg",
        "land",
        "lor",
        "implies",
        "iff",
        "to",
        "rightarrow",
        "leftarrow",
        "leftrightarrow",
        "Rightarrow",
        "Leftarrow",
        "Leftrightarrow",
        "mapsto",
        "langle",
        "rangle",
        "lfloor",
        "rfloor",
        "lceil",
        "rceil",
        "left",
        "right",
        "big",
        "Big",
        "bigg",
        "Bigg",
        # structures / styling
        "frac",
        "dfrac",
        "tfrac",
        "sqrt",
        "overline",
        "underline",
        "hat",
        "widehat",
        "bar",
        "tilde",
        "widetilde",
        "vec",
        "dot",
        "ddot",
        "boldsymbol",
        "mathbf",
        "mathrm",
        "mathcal",
        "mathbb",
        "mathfrak",
        "mathsf",
        "mathtt",
        "mathit",
        "text",
        "operatorname",
        "begin",
        "end",
        "array",
        "aligned",
        "matrix",
        "bmatrix",
        "pmatrix",
        "vmatrix",
        "cases",
        "substack",
        "quad",
        "qquad",
        "label",
        "nonumber",
        "prime",
        "ldotp",
        "colon",
    }
)

# A bare "|" not already escaped or part of \vert/\mid/\| etc.
_BARE_PIPE = re.compile(r"(?<!\\)\|")
# Single-$ inline math: a $...$ that is not part of a $$ display block. Used only
# to warn, never to rewrite (could be currency).
_SINGLE_DOLLAR = re.compile(r"(?<!\$)\$(?!\$)(?!\s)[^$\n]+?\$(?!\$)")
# Markdown links: capture the target. Skip image links (handled elsewhere).
_MD_LINK = re.compile(r"(?<!\!)\[(?:[^\]]*)\]\(([^)]*)\)")
# LaTeX control words (\foo); \\ and single-char controls (\,, \{) are ignored.
_MACRO = re.compile(r"\\([a-zA-Z]+)")
# A heading line that needs a blank line before it.
_HEADING = re.compile(r"^#{1,6}[ \t]")


def _classify_regions(markdown: str) -> list[tuple[int, int, str]]:
    """Return (start, end, kind) for every code/math span, kind in
    {fence, icode, display, inline}. Used to know which offsets are 'protected'
    (code) or 'math' so prose-only checks skip them."""
    return [
        (m.start(), m.end(), m.lastgroup or "")
        for m in SEGMENT_PATTERN.finditer(markdown)
    ]


def _fix_headings(markdown: str) -> str:
    """Insert a blank line before any heading glued to the previous line.

    kramdown only recognises a heading when it is preceded by a blank line; a
    heading stuck to the previous paragraph gets absorbed into it. Fenced code
    blocks are skipped so a shell comment like ``# do this`` is never touched.
    """
    fences = [(s, e) for s, e, k in _classify_regions(markdown) if k == "fence"]

    def in_fence(pos: int) -> bool:
        return any(s <= pos < e for s, e in fences)

    lines = markdown.split("\n")
    out: list[str] = []
    offset = 0
    for line in lines:
        line_start = offset
        offset += len(line) + 1  # +1 for the '\n'
        if (
            _HEADING.match(line)
            and not in_fence(line_start)
            and out
            and out[-1].strip() != ""
        ):
            out.append("")
        out.append(line)
    return "\n".join(out)


def _fix_pipes_in_math(markdown: str) -> str:
    """Replace a bare ``|`` inside a math span with ``\\vert`` (identical render)
    so kramdown doesn't misread it as a table-cell delimiter."""

    def repl(match: re.Match[str]) -> str:
        if match.lastgroup in ("display", "inline"):
            return _BARE_PIPE.sub(r"\\vert ", match.group(0))
        return match.group(0)

    return SEGMENT_PATTERN.sub(repl, markdown)


def _warn_single_dollar(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    code = [(s, e) for s, e, k in regions if k in ("fence", "icode")]
    for m in _SINGLE_DOLLAR.finditer(markdown):
        if any(s <= m.start() < e for s, e in code):
            continue
        logger.warning(
            "Markdown lint: single-$ inline math %r will NOT render on the blog "
            "(use \\( ... \\)); left as-is in case it is a currency amount.",
            m.group(0)[:60],
        )


def _warn_undefined_macros(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    seen: set[str] = set()
    for ms, me, kind in regions:
        if kind not in ("display", "inline"):
            continue
        for mm in _MACRO.finditer(markdown[ms:me]):
            name = mm.group(1)
            if name not in _BASE_MATHJAX_MACROS and name not in _BLOG_EXTRA_MACROS:
                seen.add(name)
    if seen:
        logger.warning(
            "Markdown lint: math uses macro(s) not in the blog's known set %s "
            "— they may render as red raw text. Known extra macros: %s.",
            sorted(seen),
            sorted(_BLOG_EXTRA_MACROS),
        )


def _warn_non_url_links(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    code = [(s, e) for s, e, k in regions if k in ("fence", "icode")]
    for m in _MD_LINK.finditer(markdown):
        if any(s <= m.start() < e for s, e in code):
            continue
        target = m.group(1).strip()
        if not target:
            continue
        if not target.startswith(("http://", "https://", "/", "#")):
            logger.warning(
                "Markdown lint: link target %r is not an http(s) URL (nor a site-"
                "relative path) — likely a broken link; use plain prose for "
                "internal cross-references.",
                target[:80],
            )


def lint_markdown(markdown: str) -> str:
    """Auto-fix safe blog-rendering issues and warn about risky ones.

    Auto-fixed (lossless): blank line before headings, bare ``|`` -> ``\\vert``
    inside math. Warned (left untouched): single-$ inline math, macros outside
    the blog's known set, and non-URL link targets. Returns the fixed Markdown.
    """
    fixed = _fix_headings(markdown)
    fixed = _fix_pipes_in_math(fixed)

    regions = _classify_regions(fixed)
    _warn_single_dollar(fixed, regions)
    _warn_undefined_macros(fixed, regions)
    _warn_non_url_links(fixed, regions)
    return fixed
