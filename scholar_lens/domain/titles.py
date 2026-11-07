"""Whether two titles name the same paper, and whether an abstract fits a
title (index records sometimes mix up papers)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Minimum similarity between a queried and a returned title before the record
# is accepted. Lower values (or substring containment) accept "LoRA+" for
# "LoRA" or a survey for the paper it surveys.
TITLE_MATCH_THRESHOLD: float = 0.9


def _norm_title(value: str) -> str:
    return " ".join(re.sub(r"[^\w ]+", " ", value.casefold()).split())


def title_similarity(queried: str, resolved: str) -> float:
    """Similarity of two titles (case/punctuation-insensitive); 1.0 when
    they are the same title."""
    return SequenceMatcher(None, _norm_title(queried), _norm_title(resolved)).ratio()


def titles_match(queried: str | None, resolved: str | None) -> bool:
    """Whether two titles name the same paper (case/punctuation-insensitive)."""
    if not queried or not resolved:
        return False

    q, r = _norm_title(queried), _norm_title(resolved)
    if not (q and r) or not _similar(q, r):
        return False
    # Version-like tokens tell sibling papers apart even when the rest of the
    # title is identical (GPT-4 vs GPT-4o, Qwen2 vs Qwen2.5, LoRA vs LoRA+).
    if _distinguishing_tokens(queried) != _distinguishing_tokens(resolved):
        return False
    # A content word only one title has names another work even at high
    # similarity: swapped ("...Large Language Models" vs "...Protein Language
    # Models"), added ("Attention Is Not All You Need", "...: A Survey",
    # "...Revisited") or dropped. Spelling variants are not differences:
    # hyphenation ("Few-Shot" / "Few Shot"), British spelling ("Visualising")
    # or a plural.
    if _squash(q) == _squash(r):
        return True
    q_only = _title_words(queried) - _title_words(resolved)
    r_only = _title_words(resolved) - _title_words(queried)
    unpaired_q = [a for a in q_only if not any(_spelling_variant(a, b) for b in r_only)]
    unpaired_r = [b for b in r_only if not any(_spelling_variant(a, b) for a in q_only)]
    return not unpaired_q and not unpaired_r


def _similar(q: str, r: str) -> bool:
    """``title_similarity`` reaches the threshold; the cheap upper bounds of
    the ratio rule most pairs out first (pools compare hundreds of titles)."""
    matcher = SequenceMatcher(None, q, r)
    return (
        matcher.real_quick_ratio() >= TITLE_MATCH_THRESHOLD
        and matcher.quick_ratio() >= TITLE_MATCH_THRESHOLD
        and matcher.ratio() >= TITLE_MATCH_THRESHOLD
    )


# Two differing title words this similar, with the same start, are one word
# spelled two ways; a different start is a prefix ("Unsupervised", "Decoder").
_VARIANT_RATIO = 0.8


def _spelling_variant(a: str, b: str) -> bool:
    return a[:3] == b[:3] and SequenceMatcher(None, a, b).ratio() >= _VARIANT_RATIO


_DASHES = re.compile(r"[-\u2010-\u2015\u2212]")


def _squash(title: str) -> str:
    return re.sub(r"[\W_]+", "", title)


def _title_words(title: str) -> set[str]:
    return {w.removesuffix("s") for w in _content_words(_DASHES.sub("", title))}


_TOKEN = re.compile(r"[\w.+]+")


def _distinguishing_tokens(title: str) -> set[str]:
    tokens = {t.strip(".").casefold() for t in _TOKEN.findall(title)}
    return {t for t in tokens if any(ch.isdigit() for ch in t) or "+" in t}


_WORD = re.compile(r"[^\W_]+")


_STOPWORDS = frozenset(
    "a an and are as at be by for from how in into is it of on or our over the "
    "to towards via what when where which why with without you your".split()
)


def _content_words(text: str) -> set[str]:
    words = set(_WORD.findall(text.casefold()))
    return {w for w in words if w not in _STOPWORDS and not w.isdigit()}


def abstract_fits_title(title: str, abstract: str | None) -> bool:
    """Whether ``abstract`` plausibly belongs to ``title``: it shares at least
    one of the title's content words (a missing abstract contradicts
    nothing). Index records sometimes pair the right title with another
    work's abstract; such an abstract shares none."""
    words = _content_words(title)
    return not (words and abstract) or bool(words & _content_words(abstract))
