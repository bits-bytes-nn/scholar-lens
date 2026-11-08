"""The post specs: each spec's kinds match its outline schema, and the
shared outline checks and evidence policy are driven by the spec's data."""

from __future__ import annotations

from typing import get_args

import pytest

from scholar_lens.domain.evidence import SourceDocument
from scholar_lens.domain.guide.evidence_guide import GuideSection
from scholar_lens.domain.paper.evidence_paper import PaperSection
from scholar_lens.domain.post_spec import (
    EvidencePolicy,
    KindCount,
    KindOrder,
    KindSpec,
    Ownership,
    PostSpec,
    evidence_policy,
    structure_problems,
)
from scholar_lens.domain.specs import (
    GUIDE_SPEC,
    PAPER_REVIEW_SPEC,
    PAPER_SUMMARY_SPEC,
    SURVEY_SPEC,
)
from scholar_lens.domain.survey.evidence_post import OutlineSection


@pytest.mark.parametrize(
    ("spec", "section"),
    [
        (GUIDE_SPEC, GuideSection),
        (PAPER_REVIEW_SPEC, PaperSection),
        (PAPER_SUMMARY_SPEC, PaperSection),
        (SURVEY_SPEC, OutlineSection),
    ],
)
def test_a_spec_names_exactly_its_outline_kinds(spec: PostSpec, section: type) -> None:
    kinds = set(get_args(section.model_fields["kind"].annotation))
    assert set(spec.ranks) == kinds
    named = {c.kind for c in spec.outline_checks if isinstance(c, KindCount)}
    for kinds_of_spec in (
        named,
        set(spec.fixed_titles),
        spec.depth_kinds,
        spec.runnable_kinds,
        set(spec.attachments),
    ):
        assert kinds_of_spec <= kinds
    if spec.tldr is not None:
        assert spec.tldr.kind in kinds


_SPEC = PostSpec(
    outline_name="outline",
    kinds=(KindSpec("intro", 0), KindSpec("body", 1), KindSpec("end", 2)),
    outline_checks=(
        KindCount("intro", "exactly_one", "one intro first", position="first"),
        KindCount("end", "at_most_one", "at most one end"),
        KindOrder("intro, body, end"),
    ),
    ownership=Ownership(),
    evidence=EvidencePolicy(),
    retry_hint="",
)


def test_outline_checks_report_in_the_spec_order() -> None:
    assert structure_problems(_SPEC, ["intro", "body", "end"]) == []
    assert structure_problems(_SPEC, ["body", "intro", "end", "end"]) == [
        "one intro first",
        "at most one end",
        "intro, body, end",
    ]


def _page(authority: str) -> SourceDocument:
    return SourceDocument.from_markdown(
        "P", "Page", "full_text", "## A\n\nText.", authority=authority  # type: ignore[arg-type]
    )


def test_the_evidence_policy_reads_kinds_fields_and_authorities() -> None:
    survey = SURVEY_SPEC.evidence
    assert evidence_policy(survey, _page(""), "code", {})
    assert evidence_policy(survey, _page(""), "number", {"cell": ""})
    assert evidence_policy(survey, _page(""), "number", {"cell": "61% 승률"}) == []
    guide = GUIDE_SPEC.evidence
    blog = _page("third_party")
    # Settings and identifiers belong to the official docs, when there are any.
    assert evidence_policy(guide, blog, "setting", {}, corpus={"official"})
    assert evidence_policy(guide, blog, "setting", {}, corpus={"reference"}) == []
    named = {"claim": "`max_batch_size` caps a batch"}
    assert evidence_policy(guide, blog, "finding", named, corpus={"official"})
    assert evidence_policy(guide, blog, "relation", named, corpus={"official"}) == []
