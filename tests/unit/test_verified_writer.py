"""Properties of the writer's claim check, verdict cache and unsupported
sentence handling (docs/engine.md, 4.2, 4.5 and 4.10)."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

import pytest

from scholar_lens.application.evidence_writer import (
    ClaimCheck,
    Edit,
    EvidenceWriter,
    FixedSection,
    PostFinding,
    SectionEdits,
    SectionReview,
    SentenceVerdict,
)
from scholar_lens.application.survey_engine import SURVEY_PROMPTS
from scholar_lens.domain.evidence import Evidence, EvidenceLedger
from scholar_lens.domain.post_spec import Limits
from scholar_lens.domain.verifiers import Problem, VerifierSet, no_checks

_ITEM = re.compile(r"^(\d+)\. (.*)$", re.M)


def _ledger() -> EvidenceLedger:
    return EvidenceLedger(
        items=[
            Evidence(
                id="E1",
                paper_id="P",
                kind="method",
                claim="캐시를 씁니다",
                quote="it uses a cache",
                anchor="S1",
            )
        ]
    )


class Model:
    """Answers the writer's calls: the claim check marks every item that
    ``bad`` accepts as unsupported; a fix answers ``edits`` (none by
    default) and the whole section as it was."""

    def __init__(
        self,
        bad: Callable[[str], bool] = lambda _s: False,
        edits: list[Edit] | None = None,
        fail_claims: bool = False,
    ) -> None:
        self.bad = bad
        self.edits = edits or []
        self.fail_claims = fail_claims
        self.sent: list[list[str]] = []
        self.prompts: list[str] = []

    async def structured(self, *, schema: Any, prompt: str, **_: Any) -> Any:
        if schema is ClaimCheck:
            if self.fail_claims:
                raise ValueError("No valid ClaimCheck (stop reason content_filtered)")
            self.prompts.append(prompt)
            items = _ITEM.findall(prompt)
            self.sent.append([s for _, s in items])
            return ClaimCheck(
                verdicts=[
                    SentenceVerdict(n=int(n), verdict="unsupported", reason="r")
                    for n, s in items
                    if self.bad(s)
                ]
            )
        if schema is SectionEdits:
            return SectionEdits(edits=self.edits)
        if schema is FixedSection:
            section = prompt.split("<section>\n", 1)[1].split("\n</section>", 1)[0]
            return FixedSection(section=section)
        if schema is SectionReview:
            return SectionReview()
        raise AssertionError(schema)


def _writer(model: Model) -> EvidenceWriter:
    return EvidenceWriter(model, SURVEY_PROMPTS, max_fix_rounds=1, limits=Limits())  # type: ignore[arg-type]


def _sentences(n: int) -> list[str]:
    return [f"문장 {chr(0xAC00 + k)}은 캐시를 씁니다 [[E1]]." for k in range(n)]


def _section(sentences: list[str], *extra: str) -> str:
    return "## 캐시\n\n" + "\n\n".join([*sentences, *extra]) + "\n"


# The section the scripted checks register under (by object, not title).
SECTION = "캐시"


def _verify(writer: EvidenceWriter, draft: str, **kwargs: Any) -> str:
    return asyncio.run(
        writer.verify_and_fix(
            draft,
            title="캐시",
            review_context="",
            verifiers=kwargs.pop("verifiers", no_checks()),
            ledger=_ledger(),
            own=list(_ledger()),
            earlier=[],
            free="",
            section=kwargs.pop("section", SECTION),
            **kwargs,
        )
    )


# ---- the verdict cache -----------------------------------------------------------


def test_cached_verdicts_are_reused_and_only_changed_sentences_are_sent() -> None:
    model = Model()
    writer = _writer(model)
    sentences = _sentences(5)
    draft = _section(sentences)
    asyncio.run(writer.check_claims(draft, _ledger()))
    asyncio.run(writer.check_claims(draft, _ledger()))
    assert len(model.sent) == 1 and model.sent[0] == sentences
    changed = draft.replace(sentences[2], "바뀐 문장은 캐시를 씁니다 [[E1]].")
    asyncio.run(writer.check_claims(changed, _ledger()))
    # The changed sentence, and the one after it (its context changed).
    assert model.sent[1] == ["바뀐 문장은 캐시를 씁니다 [[E1]].", sentences[3]]
    assert f"follows: {sentences[1]}" in model.prompts[1]


def test_a_cached_verdict_holds_only_for_its_evidence_context() -> None:
    model = Model()
    writer = _writer(model)
    draft = _section(_sentences(2))
    asyncio.run(writer.check_claims(draft, _ledger()))
    asyncio.run(writer.check_claims(draft, _ledger(), list(_ledger())))
    asyncio.run(writer.check_claims(draft, _ledger(), reference="passage"))
    assert len(model.sent) == 3


def test_a_cached_unsupported_verdict_is_merged_back() -> None:
    sentences = _sentences(3)
    model = Model(bad=lambda s: s == sentences[1])
    writer = _writer(model)
    draft = _section(sentences)
    first = asyncio.run(writer.check_claims(draft, _ledger()))
    again = asyncio.run(writer.check_claims(draft, _ledger()))
    assert first == again and [s for s, _ in again] == [sentences[1]]
    assert len(model.sent) == 1


# ---- a check that cannot run -------------------------------------------------------


def test_a_failed_claim_check_blocks_the_post_and_is_not_cached() -> None:
    model = Model(fail_claims=True)
    writer = _writer(model)
    draft = _section(_sentences(2))
    assert asyncio.run(writer.check_claims(draft, _ledger())) == []
    assert len(writer.blocking) == 1 and "unchecked" in writer.blocking[0]
    model.fail_claims = False
    asyncio.run(writer.check_claims(draft, _ledger()))
    assert len(model.sent) == 1


# ---- unsupported sentences -------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 8])
def test_a_sentence_still_unsupported_is_deleted(n: int) -> None:
    sentences = _sentences(n)
    target = sentences[1]
    writer = _writer(Model(bad=lambda s: s == target))
    out = _verify(writer, _section(sentences))
    assert target not in out
    assert all(s in out for s in sentences if s != target)
    assert not writer.blocking


@pytest.mark.parametrize(
    "exempt,judgement",
    [
        ("| 캐시 | 켜짐 [[E1]] |", False),
        ("가상의 캐시라면 요청이 줄어듭니다 [[EX]].", False),
        ("이 설계는 단순해서 고르기 쉽습니다 [[E1]].", True),
    ],
)
def test_exempt_sentences_are_listed_not_deleted(exempt: str, judgement: bool) -> None:
    sentences = _sentences(5)
    table = f"| 항목 | 값 |\n| --- | --- |\n{exempt}" if exempt.startswith("|") else ""
    body = [*sentences, table] if table else [*sentences, exempt]
    writer = _writer(Model(bad=lambda s: s == exempt))
    out = _verify(writer, _section(body), judgement=judgement)
    assert exempt in out
    assert any("may not support" in n and exempt[:20] in n for n in writer.notes)


@pytest.mark.parametrize("extra", [1, None])
def test_deleting_more_than_the_drop_share_blocks_instead(extra: int | None) -> None:
    # Just over the limit, and every sentence flagged.
    sentences = _sentences(6)
    over = int(len(sentences) * Limits().max_drop_share) + (extra or len(sentences))
    flagged = set(sentences[:over])
    writer = _writer(Model(bad=lambda s: s in flagged))
    draft = _section(sentences)
    out = _verify(writer, draft)
    assert all(s in out for s in sentences)
    assert len(writer.blocking) == 1 and "none was deleted" in writer.blocking[0]


# ---- the whole-post fix ------------------------------------------------------------------


def _subheadings(markdown: str) -> list[Problem]:
    return [Problem("subheading", "no ### headings")] if "\n### " in markdown else []


def test_a_whole_post_fix_runs_the_sections_verifier_set() -> None:
    sentences = _sentences(5)
    draft = _section(sentences)
    verifiers = VerifierSet("캐시", (_subheadings,))
    model = Model()
    writer = _writer(model)
    md = _verify(writer, draft, verifiers=verifiers)
    assert writer.checked[id(SECTION)].verifiers is verifiers
    model.edits = [Edit(find=sentences[0], replace=f"{sentences[0]}\n\n### 세부")]
    finding = PostFinding(section="캐시", problem="p", fix="f")
    out = asyncio.run(
        writer.apply_findings(
            [(SECTION, md)],
            [finding],
            _ledger(),
            title_of=lambda s: s,
            own_of=lambda _s: list(_ledger()),
            reference_of=lambda _e: "",
        )
    )
    # The edit adds a problem the section's own set finds: not taken.
    assert out == [(SECTION, md)]
    assert any("left as they were" in n and "adds problems" in n for n in writer.notes)


def test_a_whole_post_fix_settles_only_the_sentences_it_adds() -> None:
    sentences = _sentences(5)
    added = "새 문장은 근거가 없습니다 [[E1]]."
    model = Model()
    writer = _writer(model)
    md = _verify(writer, _section(sentences))
    model.bad = lambda s: s == added
    model.edits = [Edit(find=sentences[0], replace=f"{sentences[0]} {added}")]
    calls = len(model.sent)
    out = asyncio.run(
        writer.apply_findings(
            [(SECTION, md)],
            [PostFinding(section="캐시", problem="p", fix="f")],
            _ledger(),
            title_of=lambda s: s,
            own_of=lambda _s: list(_ledger()),
            reference_of=lambda _e: "",
        )
    )
    assert added not in out[0][1] and all(s in out[0][1] for s in sentences)
    # The fix's claim check read the new sentence and the one after it (its
    # context changed); the others' verdicts were cached.
    assert model.sent[calls] == [added, sentences[1]]


# ---- a failed review, and the claim check's cache parts (4.10) -------------------


class _FailingReview(Model):
    async def structured(self, *, schema: Any, prompt: str, **kwargs: Any) -> Any:
        if schema is SectionReview:
            raise ValueError("No valid SectionReview (stop reason content_filtered)")
        return await super().structured(schema=schema, prompt=prompt, **kwargs)


def test_a_review_without_a_valid_answer_blocks_the_post() -> None:
    writer = _writer(_FailingReview())
    draft = _section(_sentences(2))
    assert _verify(writer, draft) == draft
    assert len(writer.blocking) == 1
    assert "review of '캐시' could not run" in writer.blocking[0]
    assert "unreviewed" in writer.blocking[0]


class _Contexts(Model):
    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[Any] = []

    async def structured(self, *, schema: Any, prompt: str, **kwargs: Any) -> Any:
        if schema is ClaimCheck:
            self.contexts.append(kwargs.get("context"))
        return await super().structured(schema=schema, prompt=prompt, **kwargs)


def _earlier() -> Evidence:
    return Evidence(
        id="E9",
        paper_id="P",
        kind="finding",
        claim="앞 절의 사실입니다",
        quote="an earlier fact",
        anchor="S2",
    )


def test_the_claim_check_context_puts_earlier_evidence_first() -> None:
    model = _Contexts()
    writer = _writer(model)
    draft = _section(_sentences(1))
    asyncio.run(
        writer.check_claims(
            draft, _ledger(), list(_ledger()), [_earlier()], reference="passage"
        )
    )
    ((earlier, section),) = model.contexts
    # Two parts, each cached on its own: what only grows across sections,
    # then this section's evidence and its source passages.
    assert earlier.startswith("<earlier_evidence>") and "[E9]" in earlier
    assert "[E1]" not in earlier
    assert section.startswith("<section_evidence>") and "[E1]" in section
    assert "<source_sections>\npassage" in section


def test_the_first_section_has_no_earlier_part() -> None:
    model = _Contexts()
    writer = _writer(model)
    asyncio.run(
        writer.check_claims(_section(_sentences(1)), _ledger(), list(_ledger()))
    )
    ((earlier, section),) = model.contexts
    assert earlier == "" and "[E1]" in section
