"""The survey as data: the introduction, a section per lineage, the
cross-cutting themes, the comparison, insights and the conclusion, with the
tables code builds."""

from __future__ import annotations

from scholar_lens.domain.post_spec import (
    CoverageSpec,
    EvidencePolicy,
    KindCount,
    KindOrder,
    KindSpec,
    Limits,
    Ownership,
    PostSpec,
    RequiredField,
    TargetKind,
    TitleLength,
)

# Sections that read the user's reference material.
REFERENCE_KINDS = frozenset({"introduction", "cross_cutting", "insights"})

SURVEY_SPEC = PostSpec(
    name="survey",
    outline_name="outline",
    kinds=(
        KindSpec("introduction", 0),
        KindSpec("lineage", 1),
        KindSpec("cross_cutting", 2),
        KindSpec("comparison", 3),
        KindSpec("insights", 4),
        KindSpec("conclusion", 5),
    ),
    outline_checks=(
        TitleLength(),
        KindCount(
            "introduction",
            "any",
            "the first section must be the introduction",
            position="first",
        ),
        KindOrder(
            "sections must go introduction, lineages, cross-cutting, comparison, "
            "insights, conclusion"
        ),
        KindCount(
            "comparison",
            "exactly_one",
            "the outline needs exactly 1 comparison section",
        ),
        KindCount(
            "insights", "exactly_one", "the outline needs exactly 1 insights section"
        ),
        KindCount("conclusion", "at_most_one", "at most one conclusion section"),
        KindCount("cross_cutting", "at_most_one", "at most one cross-cutting section"),
    ),
    ownership=Ownership(
        unknown="section {title!r} uses unknown evidence {eid}",
        twice="{eid} is given to both {first!r} and {title!r}: give each piece "
        "of evidence to one section",
    ),
    # A survey states what papers find; their code and settings are a
    # guide's evidence. Results and limitations fill the comparison table:
    # each needs its short noun-phrase cell (a claim is a sentence, not a
    # cell).
    evidence=EvidencePolicy(
        forbidden_kinds=frozenset({"code", "setting"}),
        forbidden_message="kind {kind!r} is not used in a survey; record a "
        "method instead",
        required_fields=(
            RequiredField(
                kinds=frozenset({"number", "limitation"}),
                name="cell",
                message="a {kind} needs `cell`: a short noun phrase for the "
                "comparison table (a result with its benchmark and metric, or "
                "the limitation), in the post's language and register",
            ),
        ),
    ),
    retry_hint="This is a survey of published research papers for practitioners.",
    depth_kinds=frozenset({"lineage", "cross_cutting"}),
    # The post's reading of the field and its closing view.
    judgement_kinds=frozenset({"insights", "conclusion"}),
    attachments={"introduction": "timeline", "comparison": "comparison"},
    # Every core paper and every lineage of the plan, derived in code from
    # the plan (no extraction).
    coverage=CoverageSpec(
        kinds=(
            TargetKind("paper", "a core paper of the survey"),
            TargetKind("lineage", "a lineage of the survey"),
        )
    ),
    # A core paper is read in depth; a supporting one for what it adds.
    limits=Limits(main_source_evidence=(12, 25), minor_source_evidence=(4, 8)),
)
