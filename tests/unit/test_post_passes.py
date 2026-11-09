"""The whole-post steps, coverage and the earlier-section context of the post
engine (docs/engine.md, 4.6-4.8): every spec's steps run, key targets reach
the readers and the outline, an unassigned target asks the outline again, a
TL;DR without the headline evidence is a problem, an uncited target is a
note, and the next writer sees what earlier sections taught."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from scholar_lens.application.evidence_writer import PostReview
from scholar_lens.application.guide_engine import EvidenceGuideEngine, GuideSettings
from scholar_lens.application.paper_engine import EvidencePaperEngine, PaperSettings
from scholar_lens.application.survey_engine import EngineSettings, EvidenceSurveyEngine
from scholar_lens.domain.coverage import (
    CoverageTarget,
    ExtractedTarget,
    ExtractedTargets,
    coverage_problems,
    extraction_text,
    headline_problems,
    repair_coverage,
    targets_from,
    unassigned,
    uncited,
    unsatisfiable,
)
from scholar_lens.domain.evidence import Evidence, EvidenceLedger
from scholar_lens.domain.paper.evidence_paper import PaperOutline
from scholar_lens.domain.post_context import (
    WrittenSection,
    previous_sections,
    taught_list,
)
from scholar_lens.domain.post_spec import CoverageSpec, PostSpec, TargetKind
from scholar_lens.domain.specs import (
    GUIDE_SPEC,
    PAPER_REVIEW_SPEC,
    PAPER_SUMMARY_SPEC,
    SURVEY_SPEC,
)
from scholar_lens.domain.survey.evidence_post import plan_targets
from tests.unit.test_evidence_engine import FakeDocuments, _plan
from tests.unit.test_paper_engine import ScriptedModel, _outline, _paper

ALL_SPECS = [GUIDE_SPEC, PAPER_REVIEW_SPEC, PAPER_SUMMARY_SPEC, SURVEY_SPEC]


# ---- the specs ----------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_SPECS)
def test_every_spec_lists_the_three_whole_post_steps(spec: PostSpec) -> None:
    assert spec.post_passes.names == ("consistency", "repetition", "coverage")
    assert spec.limits.previous_sections >= 1
    # A spec's code trace needs sections a reader runs.
    assert not spec.post_passes.code_trace or spec.runnable_kinds
    kinds = {k.name for k in spec.coverage.kinds}
    assert set(spec.coverage.headline_kinds) <= kinds
    low, high = spec.coverage.count
    assert 0 <= low <= high


class _Idle:
    """A model no test call reaches."""


def _engines() -> list[Any]:
    model: Any = _Idle()
    return [
        EvidenceGuideEngine(
            model,
            GuideSettings(language="Korean", title_max_length=70, reader_tag="t"),
        ),
        EvidencePaperEngine(
            model, PaperSettings(language="Korean", post="review", reader_tag="t")
        ),
        EvidencePaperEngine(
            model, PaperSettings(language="Korean", post="summary", reader_tag="t")
        ),
        EvidenceSurveyEngine(
            model,
            FakeDocuments(),
            EngineSettings(language="Korean", title_max_length=70, reader_tag="t"),
        ),
    ]


@pytest.mark.parametrize(
    "engine", _engines(), ids=["guide", "review", "summary", "survey"]
)
def test_every_kind_composes_its_reads_from_the_shared_rules(engine: Any) -> None:
    from scholar_lens.domain.prompts.rules import CONSISTENCY_CHECKS, REPETITION_CHECKS

    systems = engine.pass_systems
    assert CONSISTENCY_CHECKS in systems["consistency"]
    assert REPETITION_CHECKS in systems["repetition"]
    assert "repetition only" in systems["repetition"]
    traced = "in `trace`" in systems["consistency"]
    assert traced == engine.spec.post_passes.code_trace


def test_each_kind_keeps_its_own_checks() -> None:
    guide, review, _summary, survey = _engines()
    # The guide's install and package checks, the paper's table checks,
    # the survey's retold papers: fragments of the kind, not the engine.
    assert "install command" in guide.pass_systems["consistency"]
    assert "[sdk_source" in guide.pass_systems["consistency"]
    assert "blank" in review.pass_systems["consistency"]
    assert "install command" not in review.pass_systems["consistency"]
    assert "lineage section" in survey.pass_systems["repetition"]


# ---- coverage in code -----------------------------------------------------------


_SPEC = CoverageSpec(
    kinds=(
        TargetKind("claim", "a claim"),
        TargetKind("result", "a result", headline=1),
    ),
    count=(1, 3),
    sections=("abstract",),
)


def test_extracted_targets_are_numbered_capped_and_marked_headline() -> None:
    answer = ExtractedTargets(
        targets=[
            ExtractedTarget(statement="first result", kind="result", source="P"),
            ExtractedTarget(statement="second result", kind="result", source="?"),
            ExtractedTarget(statement="  a   claim ", kind="other"),
            ExtractedTarget(statement="past the cap", kind="claim"),
        ]
    )
    targets = targets_from(answer, _SPEC, ["P"])
    assert [(t.id, t.kind, t.headline, t.source) for t in targets] == [
        ("T1", "result", True, "P"),
        ("T2", "result", False, "P"),
        ("T3", "claim", False, "P"),
    ]
    assert targets[2].statement == "a claim"


def test_the_extraction_reads_the_named_sections() -> None:
    text = extraction_text(_SPEC, [_paper()], 10_000)
    assert text.startswith("[2305.18290] DPO")
    assert "without a reward model" in text and "PPO, which is unstable" not in text


@pytest.mark.parametrize("chars", [5, 40, 10_000])
def test_a_documents_opening_is_the_specs_share(chars: int) -> None:
    import dataclasses

    spec = dataclasses.replace(_SPEC, sections=(), opening_chars=chars)
    paper = _paper()
    opening = paper.sections[0].text[:chars]
    text = extraction_text(spec, [paper], 100_000)
    assert opening.strip()[:chars] in text
    if chars < len(paper.sections[0].text):
        assert paper.sections[0].text[: chars + 20] not in text


def _targets() -> list[CoverageTarget]:
    return [
        CoverageTarget(id="T1", statement="a", source="P"),
        CoverageTarget(id="T2", statement="b", source="P", headline=True),
    ]


def test_coverage_maps_are_repaired_and_checked_in_code() -> None:
    coverage = repair_coverage(
        {"T1": ["E9", "E2", "E2"], "T7": ["E1"]}, _targets(), {"E1", "E2"}, {"E2": "E1"}
    )
    assert coverage == {"T1": ["E1"], "T2": []}
    assert [t.id for t in unassigned(_targets(), coverage, {"E1"})] == ["T2"]
    assert [t.id for t in unassigned(_targets(), coverage, set())] == ["T1", "T2"]
    assert [
        t.id for t in uncited(_targets(), {"T1": ["E1"], "T2": ["E3"]}, ["E3"])
    ] == ["T1"]
    headline = [(_targets()[1], ["E3"])]
    assert headline_problems(headline, {"E3"}) == []
    assert "T2" in headline_problems(headline, {"E1"})[0]


def test_a_survey_derives_its_targets_from_the_plan() -> None:
    plan = _plan()
    targets = plan_targets(plan)
    core = [p.id for p in plan.papers if p.role == "core"]
    assert [t.source for t in targets] == core + [lin.id for lin in plan.lineages]
    assert {t.kind for t in targets} == {k.name for k in SURVEY_SPEC.coverage.kinds}
    # A lineage is stated by its papers' evidence.
    for lin in plan.lineages:
        (target,) = [t for t in targets if t.source == lin.id]
        assert target.stated_by() == {p.id for p in plan.papers if p.lineage == lin.id}


def _evidence_from(*sources: str) -> list[Evidence]:
    return [
        Evidence(
            id=f"E{n}", paper_id=s, kind="method", claim="c", quote="q", anchor="S1"
        )
        for n, s in enumerate(sources, 1)
    ]


def test_targets_no_evidence_can_state_are_not_outline_problems() -> None:
    """Property, over generated targets and ledgers: with the ledger, the
    coverage problems are exactly the unassigned targets some source of
    which has evidence; the others are :func:`unsatisfiable`."""
    rng = random.Random(7)
    names = ["A", "B", "C", "D"]
    for _ in range(200):
        targets = [
            CoverageTarget(
                id=f"T{n}",
                statement=f"s{n}",
                source=rng.choice(names),
                evidence_sources=rng.sample(names, rng.randint(0, 2)),
            )
            for n in range(1, rng.randint(1, 6))
        ]
        ledger = _evidence_from(*rng.sample(names, rng.randint(0, 3)))
        coverage = {t.id: [e.id for e in ledger if rng.random() < 0.3] for t in targets}
        used = {e.id for e in ledger if rng.random() < 0.7}
        hopeless = {t.id for t in unsatisfiable(targets, ledger)}
        have = {e.paper_id for e in ledger}
        assert hopeless == {t.id for t in targets if not t.stated_by() & have}
        expected = [
            t.id for t in unassigned(targets, coverage, used) if t.id not in hopeless
        ]
        problems = coverage_problems(targets, coverage, used, ledger)
        assert [p.split()[2] for p in problems] == expected
        # Without the ledger, every unassigned target is a problem.
        assert len(coverage_problems(targets, coverage, used)) == len(
            unassigned(targets, coverage, used)
        )


# ---- the earlier-section context ------------------------------------------------


def _ledger() -> EvidenceLedger:
    return EvidenceLedger(
        items=[
            Evidence(
                id=f"E{n}",
                paper_id="P",
                kind="method",
                claim=f"claim {n}",
                quote="q",
                anchor="S1",
            )
            for n in (1, 2, 3)
        ]
    )


def test_the_taught_list_covers_every_earlier_section() -> None:
    code = "```python\ndef load(path: str) -> dict:\n    return {}\n```\n"
    sections = [
        WrittenSection(
            "A", "## 첫째\n\n첫 문장입니다 [[E1]]. 둘째 문장입니다 [[E1, E2]].\n"
        ),
        WrittenSection("B", "## 둘째\n\n코드입니다 [[E3]].\n\n" + code, runnable=True),
        WrittenSection("C", "## 셋째\n\n" + code),
    ]
    taught = taught_list(sections, _ledger(), claim_chars=5)
    assert taught.split("\n") == [
        "- 첫째",
        "  - [E1] claim",
        "  - [E2] claim",
        "- 둘째",
        "  - [E3] claim",
        "  - code: def load(path: str) -> dict",
        "- 셋째",
    ]
    assert previous_sections(sections, 1) == sections[-1].markdown.strip()
    assert previous_sections(sections, 2).startswith("## 둘째")
    assert previous_sections(sections, 0) == ""


# ---- scripted runs on the paper engine ------------------------------------------


class CoverageModel(ScriptedModel):
    """The paper script with two key targets (a claim and a headline
    result); the outline maps them from its ``attempt``-th answer on."""

    def __init__(
        self,
        post: str,
        *,
        mapped_from: int = 1,
        coverage: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__(post)
        self.mapped_from = mapped_from
        self.coverage = (
            coverage if coverage is not None else {"T1": ["E2"], "T2": ["E3"]}
        )
        self.outline_prompts: list[str] = []
        self.read_prompts: list[str] = []
        self.write_prompts: list[str] = []
        self.fix_prompts: list[str] = []

    async def run_agent(self, *, role, system, prompt, tools, limits):  # type: ignore[no-untyped-def]
        self.read_prompts.append(prompt)
        return await super().run_agent(
            role=role, system=system, prompt=prompt, tools=tools, limits=limits
        )

    async def structured(self, *, role, system, prompt, schema, context=""):  # type: ignore[no-untyped-def]
        if schema is ExtractedTargets:
            self.calls.append("ExtractedTargets")
            return ExtractedTargets(
                targets=[
                    ExtractedTarget(
                        statement="DPO는 보상 모델 없이 선호를 최적화합니다",
                        kind="claim",
                        source="2305.18290",
                    ),
                    ExtractedTarget(
                        statement="TL;DR 요약에서 PPO 대비 승률 61%",
                        kind="result",
                        source="2305.18290",
                    ),
                ]
            )
        if schema is PaperOutline:
            self.outline_prompts.append(prompt)
            outline = await super().structured(
                role=role, system=system, prompt=prompt, schema=schema, context=context
            )
            if len(self.outline_prompts) >= self.mapped_from:
                outline = outline.model_copy(update={"coverage": self.coverage})
            return outline
        return await super().structured(
            role=role, system=system, prompt=prompt, schema=schema, context=context
        )

    async def text(self, *, role, system, prompt, context="") -> str:  # type: ignore[no-untyped-def]
        if "<problems>" in prompt:
            self.fix_prompts.append(prompt)
            # The TL;DR fixed to state the headline result.
            return (
                "## TL;DR\n\nDPO는 보상 모델 없이 선호를 최적화해 PPO 대비 61% "
                "승률을 기록했습니다 [[E2, E3]].\n"
            )
        self.write_prompts.append(prompt)
        return await super().text(
            role=role, system=system, prompt=prompt, context=context
        )


def _run(model: CoverageModel, post: str = "review", **settings: Any) -> Any:
    engine = EvidencePaperEngine(
        model,
        PaperSettings(
            language="Korean",
            post=post,  # type: ignore[arg-type]
            reader_tag="sonnet",
            min_total_chars=1_000,
            max_total_chars=2_000,
            **settings,
        ),
    )
    return engine, asyncio.run(
        engine.write("DPO", _paper(), abstract="We propose DPO.")
    )


def test_the_readers_and_the_outline_get_the_key_targets() -> None:
    model = CoverageModel("review")
    engine, result = _run(model)
    # Extracted before reading, then shown to every reader and the outline.
    assert model.calls.index("ExtractedTargets") < model.calls.index("read")
    assert all(
        "<key_targets>" in p and "- T2 (result, headline" in p
        for p in model.read_prompts
    )
    assert "<key_targets>" in model.outline_prompts[0]
    assert len(model.outline_prompts) == 1
    assert result.outline.coverage == {"T1": ["E2"], "T2": ["E3"]}
    assert not any("Key targets" in n for n in result.notes)


def test_an_unassigned_target_asks_for_the_outline_again() -> None:
    model = CoverageModel("review", mapped_from=2)
    _, result = _run(model)
    assert len(model.outline_prompts) == 2
    assert "<previous_problems>" in model.outline_prompts[1]
    assert "key target T1" in model.outline_prompts[1]
    assert result.outline.coverage["T2"] == ["E3"]


def test_a_target_still_unassigned_on_the_last_attempt_is_a_note() -> None:
    model = CoverageModel("review", mapped_from=99)
    engine, result = _run(model, outline_repairs=1)
    assert len(model.outline_prompts) == 2
    assert any("Key targets no section of the outline" in n for n in result.notes)
    assert not result.blocking


def test_a_later_outline_that_breaks_a_rule_falls_back_to_the_best_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attempt that maps no targets but is sound, then repairs that
    break a structure rule: the sound outline is kept, with the note,
    instead of a failed run."""
    model = CoverageModel("review", mapped_from=99)
    original = EvidencePaperEngine.outline_problems
    seen: list[int] = []

    def broken_after_first(self: Any, outline: Any, ledger: Any) -> list[str]:
        seen.append(1)
        found = original(self, outline, ledger)
        return found if len(seen) == 1 else [*found, "a section breaks a rule"]

    monkeypatch.setattr(EvidencePaperEngine, "outline_problems", broken_after_first)
    _, result = _run(model, outline_repairs=2)
    assert len(model.outline_prompts) == 3
    assert any("Key targets no section of the outline" in n for n in result.notes)
    assert result.outline.sections


class AttemptsModel(CoverageModel):
    """The outline maps ``coverages[k]`` on its k-th answer."""

    def __init__(self, coverages: list[dict[str, list[str]]]) -> None:
        super().__init__("review", mapped_from=99)
        self.coverages = coverages

    async def structured(self, *, role, system, prompt, schema, context=""):  # type: ignore[no-untyped-def]
        outline = await super().structured(
            role=role, system=system, prompt=prompt, schema=schema, context=context
        )
        if schema is PaperOutline:
            coverage = self.coverages[len(self.outline_prompts) - 1]
            outline = outline.model_copy(update={"coverage": coverage})
        return outline


ONE = {"T1": ["E2"]}
OTHER = {"T2": ["E3"]}
NONE: dict[str, list[str]] = {}


@pytest.mark.parametrize(
    ("coverages", "broken", "kept"),
    [
        # Sound and better, then sound and worse, then broken.
        ([ONE, NONE, NONE], 3, ONE),
        # Sound and worse, then sound and better, then broken.
        ([NONE, ONE, NONE], 3, ONE),
        # Sound and better, broken, then the last sound but worse.
        ([ONE, NONE, NONE], 2, ONE),
        # A tie keeps the later one; the last attempt sound.
        ([NONE, NONE, NONE], 2, NONE),
        # Two sound attempts that map as many targets, then a broken one:
        # the later of the two is kept.
        ([ONE, OTHER, NONE], 3, OTHER),
    ],
)
def test_the_outline_kept_is_the_sound_one_that_maps_the_most_targets(
    monkeypatch: pytest.MonkeyPatch,
    coverages: list[dict[str, list[str]]],
    broken: int,
    kept: dict[str, list[str]],
) -> None:
    model = AttemptsModel(coverages)
    original = EvidencePaperEngine.outline_problems
    seen: list[int] = []

    def breaks_one(self: Any, outline: Any, ledger: Any) -> list[str]:
        seen.append(1)
        found = original(self, outline, ledger)
        return found if len(seen) != broken else [*found, "a section breaks a rule"]

    monkeypatch.setattr(EvidencePaperEngine, "outline_problems", breaks_one)
    _, result = _run(model, outline_repairs=2)
    assert len(model.outline_prompts) == 3
    assert {t: ids for t, ids in result.outline.coverage.items() if ids} == kept
    assert any("Key targets no section of the outline" in n for n in result.notes)


def test_a_target_whose_source_gave_no_evidence_is_a_note_not_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target only an unread source states cannot be mapped: no outline
    is asked again for it, and the PR note says why."""
    model = CoverageModel("review")
    original = EvidencePaperEngine.key_targets

    async def with_unread(self: Any, documents: Any) -> list[CoverageTarget]:
        found = await original(self, documents)
        return [
            *found,
            CoverageTarget(id="T9", statement="unread", source="no-such-source"),
        ]

    monkeypatch.setattr(EvidencePaperEngine, "key_targets", with_unread)
    _, result = _run(model, outline_repairs=2)
    assert len(model.outline_prompts) == 1
    assert any("sources gave no evidence" in n and "T9" in n for n in result.notes)
    assert not result.blocking


def test_a_tldr_without_the_headline_evidence_is_fixed() -> None:
    # The summary's scripted TL;DR cites only E2; the headline result is E3.
    model = CoverageModel("summary")
    _, result = _run(model, "summary")
    (fix,) = [p for p in model.fix_prompts if "does not state the key target T2" in p]
    assert "## TL;DR" in fix
    tldr = dict(result.sections)["tldr"]
    assert "61%" in tldr
    tldr_prompt = next(
        p for p in model.write_prompts if "Write the summary's TL;DR" in p
    )
    assert "<headline_targets>" in tldr_prompt and "T2" in tldr_prompt


def test_a_target_the_body_never_cites_is_a_note_not_a_block() -> None:
    # T1 is mapped to evidence a section has but the body never cites.
    model = CoverageModel("review", coverage={"T1": ["E1", "E2"], "T2": ["E3"]})
    engine, _ = _run(model)
    outline = _outline(coverage={"T1": ["E1"], "T2": ["E3"]})
    engine.notes.clear()
    written = [
        (outline.sections[1], "## 보상 모델이 만드는 불안정\n\n불안정합니다.\n"),
        (
            outline.sections[3],
            "## 승률\n\nDPO는 PPO 대비 61% 승률을 기록했습니다 [[E3]].\n",
        ),
    ]
    engine.coverage_notes(outline, written)
    (note,) = engine.notes
    assert "T1" in note and "T2" not in note
    assert not engine.blocking


def test_the_next_writer_sees_the_taught_list_and_the_last_section_only() -> None:
    model = CoverageModel("review")
    _run(model)
    prompt = next(p for p in model.write_prompts if "kind: experiments" in p)
    taught = prompt.split("<already_taught>\n", 1)[1].split("\n</already_taught>", 1)[0]
    previous = prompt.split("<previous_sections>\n", 1)[1].split(
        "\n</previous_sections>", 1
    )[0]
    # Every earlier section in the list, with the claims it cited ...
    assert "- 보상 모델이 만드는 불안정" in taught and "[E1]" in taught
    assert "[E2]" in taught
    # ... and only the last one (limits.previous_sections = 1) in full.
    assert "DPO는 보상 모델 없이 선호를 최적화합니다" in previous
    assert "RLHF는" not in previous
    assert "<post_so_far>" not in prompt


class FailingReads(CoverageModel):
    async def structured(self, *, role, system, prompt, schema, context=""):  # type: ignore[no-untyped-def]
        if schema is PostReview and "repetition only" in system:
            raise ValueError("No valid PostReview (stop reason content_filtered)")
        return await super().structured(
            role=role, system=system, prompt=prompt, schema=schema, context=context
        )


def test_a_whole_post_read_without_an_answer_blocks_the_post() -> None:
    model = FailingReads("review")
    _, result = _run(model)
    (problem,) = result.blocking
    assert "whole-post repetition review could not run" in problem
    assert "consistency" in model.calls


def test_a_source_section_the_outline_leaves_out_asks_again_then_is_noted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scholar_lens.application import post_engine

    gap = "source section S2 'Experiments' of 2305.18290 has 2 evidence items"
    seen: list[tuple[str, ...]] = []

    def gaps(documents, ledger, used, same):  # type: ignore[no-untyped-def]
        seen.append(tuple(d.paper_id for d in documents))
        return [gap] if len(seen) == 1 else []

    monkeypatch.setattr(post_engine, "unused_source_sections", gaps)
    model = CoverageModel("review")
    _, result = _run(model)
    # A review covers its paper (not the code summary): asked once more.
    assert seen[0] == ("2305.18290",)
    assert len(model.outline_prompts) == 2 and gap in model.outline_prompts[1]

    # Still left out on the last attempt: noted, not a failed run.
    monkeypatch.setattr(post_engine, "unused_source_sections", lambda *a: [gap])
    model = CoverageModel("review")
    _, result = _run(model, outline_repairs=1)
    assert len(model.outline_prompts) == 2
    assert any("Source sections the outline leaves out" in n for n in result.notes)
    assert not result.blocking


def test_a_summary_is_not_held_to_every_source_section() -> None:
    model = CoverageModel("summary")
    engine, _ = _run(model, post="summary")
    assert engine.covered_documents() == ()
