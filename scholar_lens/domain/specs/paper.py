"""The paper review and summary as data.

A review goes TL;DR, the problem, the method in depth, the experiments,
optional analysis, and the post's own assessment. A summary is a TL;DR
paragraph and five fixed sections.
"""

from __future__ import annotations

from typing import Literal

from scholar_lens.domain.post_spec import (
    CoverageSpec,
    EvidencePolicy,
    KindCount,
    KindOrder,
    KindSequence,
    KindSpec,
    Limits,
    MaxSections,
    OutlineCheck,
    Ownership,
    PostSpec,
    TargetKind,
    TldrSpec,
)

TLDR_TITLE = "TL;DR"
CLOSING_TITLES = {"Korean": "한계와 시사점", "": "Limitations and Takeaways"}
# A summary's five sections, in order (blog convention).
SUMMARY_SECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "Korean": (
        ("problem", "연구 배경"),
        ("method", "제안 방법"),
        ("implementation", "구현 방식"),
        ("experiments", "실험 결과"),
        ("closing", "의의와 한계"),
    ),
    "": (
        ("problem", "Background"),
        ("method", "Proposed Method"),
        ("implementation", "Implementation"),
        ("experiments", "Results"),
        ("closing", "Significance and Limitations"),
    ),
}

_KINDS = (
    KindSpec("tldr", 0),
    KindSpec("problem", 1),
    KindSpec("method", 2),
    KindSpec("implementation", 2),
    KindSpec("experiments", 3),
    KindSpec("analysis", 4),
    KindSpec("closing", 5),
)
_SHAPE: tuple[OutlineCheck, ...] = (
    KindCount(
        "tldr",
        "exactly_one",
        "the post starts with exactly one tldr section",
        position="first",
    ),
    KindCount(
        "closing",
        "exactly_one",
        "the post ends with exactly one closing section",
        position="last",
    ),
    KindCount("method", "at_least_one", "at least one method section"),
    KindCount("problem", "exactly_one", "exactly one problem section"),
    KindOrder(
        "sections must go tldr, problem, method and implementation, "
        "experiments, analysis, closing"
    ),
)
# A paper post states what the paper finds and does; code and settings are a
# guide's evidence.
_EVIDENCE = EvidencePolicy(
    forbidden_kinds=frozenset({"code", "setting"}),
    forbidden_message="kind {kind!r} is not used for a paper; record a method instead",
)
_TLDR = TldrSpec(
    kind="tldr",
    title=TLDR_TITLE,
    retry_hint="This is the TL;DR of a paper post.",
    review_context="Section kind: tldr.",
)
_RETRY_HINT = "This is a post explaining a published research paper."
# The closing section holds the post's own assessment of the paper.
_JUDGEMENT = frozenset({"closing"})
_CLAIM = TargetKind(
    "claim",
    "a key claim other than a main result: the problem and the gap it fills, "
    "the key idea, a property or limitation the paper states",
)


def _coverage(count: tuple[int, int], headline: int) -> CoverageSpec:
    """Key targets from the abstract and the introduction; the first
    ``headline`` main results are what the TL;DR must state."""
    return CoverageSpec(
        kinds=(
            _CLAIM,
            TargetKind(
                "result",
                "a main result: the paper's method against its baselines in its "
                "headline setting, with the benchmark and metric",
                headline=headline,
            ),
        ),
        count=count,
        sections=("abstract", "introduction"),
    )


PAPER_REVIEW_SPEC = PostSpec(
    name="review",
    outline_name="paper outline",
    kinds=_KINDS,
    outline_checks=(*_SHAPE, MaxSections()),
    ownership=Ownership(empty="section '{title}' has no evidence", exempt=("tldr",)),
    evidence=_EVIDENCE,
    retry_hint=_RETRY_HINT,
    tldr=_TLDR,
    fixed_titles={"closing": CLOSING_TITLES},
    depth_kinds=frozenset({"method"}),
    judgement_kinds=_JUDGEMENT,
    coverage=_coverage((5, 8), headline=2),
)

PAPER_SUMMARY_SPEC = PostSpec(
    name="summary",
    outline_name="paper outline",
    kinds=_KINDS,
    outline_checks=(
        *_SHAPE,
        KindSequence(
            kinds=tuple(k for k, _ in SUMMARY_SECTIONS[""]),
            skip=("tldr",),
            message="a summary has exactly the sections problem, method, "
            "implementation, experiments, closing",
        ),
        MaxSections(),
    ),
    # A summary's fixed sections stand even when the paper gives one none.
    ownership=Ownership(),
    evidence=_EVIDENCE,
    retry_hint=_RETRY_HINT,
    tldr=_TLDR,
    fixed_titles={
        kind: {lang: dict(titles)[kind] for lang, titles in SUMMARY_SECTIONS.items()}
        for kind, _ in SUMMARY_SECTIONS[""]
    },
    depth_kinds=frozenset({"method"}),
    judgement_kinds=_JUDGEMENT,
    coverage=_coverage((3, 5), headline=1),
    # What a five-minute summary needs.
    limits=Limits(main_source_evidence=(6, 12)),
)


def paper_spec(post: Literal["review", "summary"]) -> PostSpec:
    """The spec of a paper ``post``."""
    return PAPER_REVIEW_SPEC if post == "review" else PAPER_SUMMARY_SPEC
