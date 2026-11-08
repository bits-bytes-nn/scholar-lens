"""Properties of the writer's claim check, verdict cache and unsupported
sentence handling (docs/engine.md, 4.2, 4.5 and 4.10)."""

from __future__ import annotations

import asyncio
import itertools
import re
from collections.abc import Callable
from typing import Any

import pytest

from scholar_lens.application.evidence_writer import (
    ClaimCheck,
    Edit,
    EvidenceCheck,
    EvidenceWriter,
    FixedSection,
    PostFinding,
    ReviewIssue,
    SectionEdits,
    SectionReview,
)
from scholar_lens.application.survey_engine import SURVEY_PROMPTS
from scholar_lens.domain.evidence import Evidence, EvidenceLedger
from scholar_lens.domain.evidence_draft import prose_sentences, section_problems
from scholar_lens.domain.post_spec import Limits
from scholar_lens.domain.verifiers import Problem, VerifierSet, no_checks
from tests.fakes import claim_check

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
        # The claim checks (the first ``omit_calls``) leave out the items
        # ``omit`` accepts.
        self.omit: Callable[[str], bool] = lambda _s: False
        self.omit_calls = 0
        self.sent: list[list[str]] = []
        self.prompts: list[str] = []

    async def structured(self, *, schema: Any, prompt: str, **_: Any) -> Any:
        if schema is ClaimCheck:
            if self.fail_claims:
                raise ValueError("No valid ClaimCheck (stop reason content_filtered)")
            self.prompts.append(prompt)
            items = _ITEM.findall(prompt)
            self.sent.append([s for _, s in items])
            check = claim_check(
                prompt, lambda _n, s: ("unsupported", "r") if self.bad(s) else None
            )
            if len(self.sent) <= self.omit_calls:
                # An answer that leaves items out.
                left_out = {int(n) for n, s in items if self.omit(s)}
                check.verdicts = [v for v in check.verdicts if v.n not in left_out]
            return check
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


@pytest.mark.parametrize("bad", [False, True])
def test_items_an_answer_leaves_out_are_asked_again_alone(bad: bool) -> None:
    """An omitted item is unchecked, not supported: it is not cached from
    the first answer, it is asked again on its own, and its verdict then
    counts."""
    sentences = _sentences(4)
    model = Model(bad=lambda s: bad and s == sentences[2])
    model.omit, model.omit_calls = (lambda s: s == sentences[2]), 1
    writer = _writer(model)
    found = asyncio.run(writer.check_claims(_section(sentences), _ledger()))
    assert model.sent == [sentences, [sentences[2]]]
    assert [s for s, _ in found] == ([sentences[2]] if bad else [])
    assert not writer.blocking
    # Both answers are cached now.
    asyncio.run(writer.check_claims(_section(sentences), _ledger()))
    assert len(model.sent) == 2


def test_items_still_left_out_are_not_cached() -> None:
    sentences = _sentences(3)
    model = Model()
    model.omit, model.omit_calls = (lambda s: s == sentences[0]), 2
    writer = _writer(model)
    asyncio.run(writer.check_claims(_section(sentences), _ledger()))
    assert model.sent == [sentences, [sentences[0]]]
    # The others were cached; the omitted one is asked on the next check.
    asyncio.run(writer.check_claims(_section(sentences), _ledger()))
    assert model.sent[2] == [sentences[0]]


def _skipped_then_fixed(skips: int) -> tuple[Model, list[str]]:
    """A section whose first unit the claim check leaves out of its first
    ``skips`` calls, and whose last unit is unsupported until a fix
    rewrites it (so the fix loop runs a second round)."""
    sentences = _sentences(6)
    bad = sentences[-1]
    model = Model(bad=lambda s: s == bad)
    model.omit, model.omit_calls = (lambda s: s == sentences[0]), skips
    model.edits = [Edit(find=bad, replace="고친 문장은 캐시를 씁니다 [[E1]].")]
    return model, sentences


def test_an_item_left_out_in_one_round_and_judged_in_the_next_does_not_block() -> None:
    # Left out of both asks of round 0, judged in round 1.
    model, sentences = _skipped_then_fixed(skips=2)
    writer = _writer(model)
    out = _verify(writer, _section(sentences))
    assert model.sent[1] == [sentences[0]] and sentences[0] in model.sent[2]
    assert sentences[0] in out
    assert not writer.blocking


def test_an_item_never_judged_blocks_the_post() -> None:
    model, sentences = _skipped_then_fixed(skips=99)
    writer = _writer(model)
    out = _verify(writer, _section(sentences))
    assert sentences[0] in out
    (blocking,) = writer.blocking
    assert "no verdict for 1 of its items" in blocking
    assert sentences[0][:20] in blocking


@pytest.mark.parametrize("replace", ["", "다시 쓴 문장은 캐시를 씁니다 [[E1]]."])
def test_an_unjudged_item_a_fix_rewrites_or_deletes_does_not_block(
    replace: str,
) -> None:
    model, sentences = _skipped_then_fixed(skips=99)
    model.edits = [Edit(find=sentences[0], replace=replace)]
    writer = _writer(model)
    out = _verify(writer, _section(sentences))
    assert sentences[0] not in out
    assert not writer.blocking


def test_a_replayed_run_blocks_on_unchecked_items_as_an_uninterrupted_one() -> None:
    """Stage 1 leaves an item unjudged; stage 2 checks the section again
    with a claim check that cannot run (the item stays unchecked), stage 3
    with one that judges it (the block is lifted). A writer that replays
    the saved stages blocks exactly as the one that ran them."""
    sentences = _sentences(2)
    draft = _section(sentences)
    model = Model()
    model.omit_calls = 99
    whole, resumed = _writer(model), _writer(Model())
    states = []
    for stage in ("skip", "fail", "judge"):
        model.fail_claims = stage == "fail"
        model.omit = lambda s, skip=stage == "skip": skip and s == sentences[0]
        mark = whole.state_mark()
        _verify(whole, draft)
        states.append(whole.state_since(mark))
        if stage == "fail":
            # The replayed writer runs this stage itself (its checkpoint
            # missed): only the replayed record keeps the item unchecked.
            resumed.model = model
            _verify(resumed, draft)
        else:
            resumed.replay_state(states[-1])
        assert resumed.blocking == whole.blocking, stage
        assert any("no verdict" in b for b in whole.blocking) == (stage != "judge")


def test_the_claim_check_schema_asks_for_a_verdict_per_item() -> None:
    described = ClaimCheck.model_json_schema()["properties"]["verdicts"]
    assert "every numbered item" in described["description"]


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


@pytest.mark.parametrize("change", ["insert", "rewrite", "delete"])
def test_a_whole_post_fix_settles_the_sentence_after_a_change(change: str) -> None:
    """The unit after a change is judged again in its new context: a new
    objection to it is settled, not cached and dropped."""
    sentences = _sentences(6)
    model = Model()
    writer = _writer(model)
    md = _verify(writer, _section(sentences))
    new = "새 문장은 캐시를 씁니다 [[E1]]."
    edit = {
        "insert": Edit(find=sentences[1], replace=f"{sentences[1]}\n\n{new}"),
        "rewrite": Edit(find=sentences[1], replace=new),
        "delete": Edit(find=sentences[1], replace=""),
    }[change]
    # Only the whole-post fix edits; the narrowing that follows does not.
    model.edits = [edit]
    original = model.structured

    async def once(**call: Any) -> Any:
        answer = await original(**call)
        if call["schema"] is SectionEdits:
            model.edits = []
        return answer

    model.structured = once  # type: ignore[method-assign]
    model.bad = lambda s: s == sentences[2]
    out = _whole_post_fix(writer, md)[0][1]
    assert sentences[2] not in out
    assert all(s in out for s in (sentences[0], sentences[3], sentences[5]))


def _foreign(markdown: str) -> list[Problem]:
    """A stand-in verifier: each line citing [[E9]] cites foreign evidence."""
    return [
        Problem("foreign_evidence", f"foreign evidence [E9] in: {line}", line)
        for line in markdown.split("\n")
        if "[[E9]]" in line
    ]


def _whole_post_fix(writer: EvidenceWriter, md: str) -> list[tuple[str, str]]:
    return asyncio.run(
        writer.apply_findings(
            [(SECTION, md)],
            [PostFinding(section="캐시", problem="p", fix="f")],
            _ledger(),
            title_of=lambda s: s,
            own_of=lambda _s: list(_ledger()),
            reference_of=lambda _e: "",
        )
    )


@pytest.mark.parametrize("more", [0, 1, 2])
def test_a_whole_post_fix_that_adds_a_problem_of_a_kind_the_section_has_is_not_taken(
    more: int,
) -> None:
    sentences = _sentences(5)
    foreign = "다른 절의 사실입니다 [[E9]]."
    verifiers = VerifierSet("캐시", (_foreign,))
    model = Model()
    writer = _writer(model)
    md = _verify(writer, _section(sentences, foreign), verifiers=verifiers)
    extra = " ".join(
        f"또 다른 사실 {chr(0xB098 + k)}입니다 [[E9]]." for k in range(more)
    )
    replace = (
        f"{sentences[0]}\n\n{extra}" if more else "고친 문장은 캐시를 씁니다 [[E1]]."
    )
    model.edits = [Edit(find=sentences[0], replace=replace)]
    writer.notes.clear()
    out = _whole_post_fix(writer, md)
    assert (out == [(SECTION, md)]) is (more > 0)
    if more:
        # The problem named is the one the fix brought, not the old one.
        assert any("adds problems" in n and "또 다른" in n for n in writer.notes)
    else:
        # The problem the fixed section still has reaches the PR notes.
        assert any("still has" in n and foreign in n for n in writer.notes)


# ---- a failed review, and the claim check's cache parts (4.10) -------------------


class _FailingReview(Model):
    async def structured(self, *, schema: Any, prompt: str, **kwargs: Any) -> Any:
        if schema is SectionReview:
            raise ValueError("No valid SectionReview (stop reason content_filtered)")
        return await super().structured(schema=schema, prompt=prompt, **kwargs)


# ---- the whole-post fix and the program a reader pastes ----------------------------

_DEFINE = "client = Client()"
_EXAMPLE = f"```python\nfrom pkg import Client\n\n{_DEFINE}\n```"
_USE = "```python\nclient.run()\n```"


def _program(written: Any) -> list[Problem]:
    from scholar_lens.domain.program import program_problems

    return program_problems([(t, md, True) for t, md in written])


@pytest.mark.parametrize("checked", [True, False])
def test_a_whole_post_fix_that_breaks_a_later_sections_code_is_not_taken(
    checked: bool,
) -> None:
    sentences = _sentences(3)
    model = Model()
    writer = _writer(model)
    md = _verify(writer, _section(sentences, _EXAMPLE))
    # The later section runs the example's client; the fix drops its
    # definition (the block still parses, the section's own checks pass).
    later = ("뒤 절", "## 뒤 절\n\n실행합니다.\n\n" + _USE + "\n")
    model.edits = [Edit(find=f"\n{_DEFINE}", replace="")]
    out = asyncio.run(
        writer.apply_findings(
            [(SECTION, md), later],
            [PostFinding(section="캐시", problem="p", fix="f")],
            _ledger(),
            title_of=lambda s: s,
            own_of=lambda _s: list(_ledger()),
            reference_of=lambda _e: "",
            post_check=_program if checked else None,
        )
    )
    assert (out[0][1] == md) == checked
    if checked:
        assert any("adds problems" in n and "client" in n for n in writer.notes)


def test_a_whole_post_fix_may_not_delete_the_first_copy_of_a_code_block() -> None:
    """A later section repeats the running example's block: the first
    copy stays (the code between them uses it); a copy in an earlier
    section lets it go."""
    sentences = _sentences(3)
    copy = ("다른 절", "## 다른 절\n\n같은 코드입니다.\n\n" + _EXAMPLE + "\n")
    for earlier in (False, True):
        model = Model()
        writer = _writer(model)
        md = _verify(writer, _section(sentences, _EXAMPLE))
        model.edits = [Edit(find=_EXAMPLE, replace="")]
        written = [copy, (SECTION, md)] if earlier else [(SECTION, md), copy]
        out = dict(
            asyncio.run(
                writer.apply_findings(
                    written,
                    [PostFinding(section="캐시", problem="p", fix="f")],
                    _ledger(),
                    title_of=lambda s: s,
                    own_of=lambda _s: list(_ledger()),
                    reference_of=lambda _e: "",
                )
            )
        )
        assert (_EXAMPLE in out[SECTION]) == (not earlier)


def test_a_review_without_a_valid_answer_blocks_the_post() -> None:
    writer = _writer(_FailingReview())
    draft = _section(_sentences(2))
    assert _verify(writer, draft) == draft
    assert len(writer.blocking) == 1
    assert "review of '캐시' could not run" in writer.blocking[0]
    assert "unreviewed" in writer.blocking[0]


class _Reviewed(Model):
    """A review that finds one problem in the first sentence."""

    def __init__(self, sentence: str, edits: list[Edit]) -> None:
        super().__init__(edits=edits)
        self.sentence = sentence

    async def structured(self, *, schema: Any, prompt: str, **kwargs: Any) -> Any:
        if schema is SectionReview:
            return SectionReview(
                issues=[
                    ReviewIssue(sentence=self.sentence, category="depth", fix="깊게")
                ]
            )
        return await super().structured(schema=schema, prompt=prompt, **kwargs)


@pytest.mark.parametrize("outcome", ["taken", "rejected", "unchanged", "unparsed"])
def test_review_problems_no_taken_fix_addressed_reach_the_notes(outcome: str) -> None:
    sentences = _sentences(5)
    replace = {
        "taken": "고친 문장은 캐시를 자세히 씁니다 [[E1]].",
        "rejected": f"{sentences[0]}\n\n### 세부",
        "unchanged": sentences[0],
        "unparsed": "",
    }[outcome]
    edits = [] if outcome == "unparsed" else [Edit(find=sentences[0], replace=replace)]
    writer = _writer(_Reviewed(sentences[0], edits))
    _verify(writer, _section(sentences), verifiers=VerifierSet("캐시", (_subheadings,)))
    noted = [n for n in writer.notes if "review problems no fix addressed" in n]
    assert bool(noted) is (outcome != "taken")
    assert all(sentences[0][:20] in n for n in noted)


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


class _FailingEvidenceCheck(Model):
    async def structured(self, *, schema: Any, prompt: str, **kwargs: Any) -> Any:
        if schema is EvidenceCheck:
            raise ValueError("No valid EvidenceCheck (stop reason max_tokens)")
        return await super().structured(schema=schema, prompt=prompt, **kwargs)


def test_an_evidence_check_without_a_valid_answer_blocks_and_drops_nothing() -> None:
    writer = _writer(_FailingEvidenceCheck())
    assert asyncio.run(writer.wrong_evidence(list(_ledger()))) == {}
    assert len(writer.blocking) == 1
    assert "evidence check could not run" in writer.blocking[0]


# ---- answer completeness of the evidence check ------------------------------------


def _items(n: int) -> list[Evidence]:
    return [
        Evidence(
            id=f"E{k}",
            paper_id="P",
            kind="method",
            claim=f"주장 {k}",
            quote=f"quote {k}",
            anchor="S1",
        )
        for k in range(1, n + 1)
    ]


class _PartialEvidenceCheck(Model):
    """Answers the evidence check for the ids ``answer(call, ids)`` keeps;
    ``wrong`` ids are wrong."""

    def __init__(
        self, answer: Callable[[int, list[str]], list[str]], wrong: set[str]
    ) -> None:
        super().__init__()
        self.answer = answer
        self.wrong = wrong
        self.asked: list[list[str]] = []

    async def structured(self, *, schema: Any, prompt: str, **kwargs: Any) -> Any:
        if schema is EvidenceCheck:
            from tests.fakes import evidence_check

            ids = re.findall(r"^\[(E\d+)\]", prompt, re.M)
            self.asked.append(ids)
            kept = set(self.answer(len(self.asked) - 1, ids))
            full = evidence_check(prompt, lambda i: "틀림" if i in self.wrong else None)
            return EvidenceCheck(verdicts=[v for v in full.verdicts if v.id in kept])
        return await super().structured(schema=schema, prompt=prompt, **kwargs)


def test_evidence_the_check_leaves_out_is_asked_again_then_blocks() -> None:
    model = _PartialEvidenceCheck(lambda call, ids: ids[:1], wrong={"E3"})
    writer = _writer(model)
    wrong = asyncio.run(writer.wrong_evidence(_items(3)))
    # Asked once for all, again only for those it left out.
    assert model.asked == [["E1", "E2", "E3"], ["E2", "E3"]]
    assert wrong == {}
    (blocking,) = writer.blocking
    assert "no verdict for 1 of its items" in blocking and "E3" in blocking


def test_evidence_answered_when_asked_again_passes_and_is_judged() -> None:
    model = _PartialEvidenceCheck(
        lambda call, ids: ids if call else ids[:1], wrong={"E3"}
    )
    writer = _writer(model)
    assert asyncio.run(writer.wrong_evidence(_items(3))) == {"E3": "틀림"}
    assert not writer.blocking and len(model.asked) == 2


@pytest.mark.parametrize("seed", range(20))
def test_complete_answers_never_passes_an_item_it_did_not_answer(seed: int) -> None:
    import random

    from scholar_lens.application.evidence_writer import complete_answers

    rng = random.Random(seed)
    items = list(range(rng.randint(0, 12)))
    asks = rng.randint(1, 3)
    calls: list[list[int]] = []

    async def ask(pending: list[int], attempt: int) -> dict[int, str]:
        calls.append(pending)
        # Answers a random subset, plus an item it was not sent.
        return {k: f"v{k}" for k in pending if rng.random() < 0.5} | {99: "x"}

    result = asyncio.run(complete_answers(items, ask, asks=asks))
    assert result.ran and len(calls) <= asks
    assert set(result.answers) | set(result.missing) == set(items)
    assert not set(result.answers) & set(result.missing)
    assert 99 not in result.answers
    # Each call after the first sends only what is still unanswered.
    for before, after in itertools.pairwise(calls):
        assert set(after) <= set(before)


# ---- a minimum length never keeps unsupported content -----------------------------


def _min_prose(chars: int) -> VerifierSet:
    from scholar_lens.domain.evidence_draft import TOO_SHORT_PROBLEM, prose_sentences

    def short(markdown: str) -> list[Problem]:
        length = len(" ".join(prose_sentences(markdown)))
        return [Problem(TOO_SHORT_PROBLEM, "too short")] if length < chars else []

    return VerifierSet("캐시", (short,))


@pytest.mark.parametrize("narrowed", [True, False])
def test_unsupported_sentences_go_even_when_the_section_becomes_short(
    narrowed: bool,
) -> None:
    """The fix loop may not cut the section under its minimum, but the
    settling of unsupported sentences (narrowed in a transaction, or deleted
    in code) is not refused by it: the section ends short, with a note."""
    sentences = _sentences(3)
    bad = sentences[1]
    draft = _section(sentences)
    model = Model(
        bad=lambda s: s == bad,
        edits=[Edit(find=f"{bad}\n\n", replace="")] if narrowed else [],
    )
    writer = _writer(model)
    verifiers = _min_prose(len(" ".join(sentences)) - 1)
    out = _verify(writer, draft, verifiers=verifiers)
    assert bad not in out and sentences[0] in out and sentences[2] in out
    assert not writer.blocking
    assert any("below its minimum length" in n for n in writer.notes)


def test_a_narrowing_that_leaves_the_section_short_is_kept_and_noted() -> None:
    """Narrowing an unsupported sentence may take the section below its
    minimum length: the narrowed sentence is kept (not deleted as if the
    narrowing failed), and the minimum length is left to a note."""
    limits = Limits()
    sentences = _sentences(30)
    bad = "이 캐시는 모든 요청을 한 번에 처리하고 지연을 없애며 메모리를 쓰지 않습니다 [[E1]]."
    narrowed = "캐시를 씁니다 [[E1]]."
    draft = _section([sentences[0], bad, *sentences[1:]])
    # The target the draft's prose just meets (at or above the floor, so
    # the minimum applies).
    target = int(len(" ".join(prose_sentences(draft))) / limits.min_length_share)
    assert target >= limits.min_length_floor

    def checked(markdown: str) -> list[Problem]:
        return section_problems(
            markdown,
            title="캐시",
            own=["E1"],
            earlier=(),
            ledger=_ledger(),
            earlier_numbers={},
            figure_urls=(),
            free_numbers="",
            target_chars=target,
            limits=limits,
        )

    verifiers = VerifierSet("캐시", (checked,))
    # The draft is at its minimum; the narrowed one is below it.
    assert checked(draft) == []
    narrowed_draft = draft.replace(bad, narrowed)
    assert [p.kind for p in checked(narrowed_draft)] == ["too_short"]
    model = Model(bad=lambda s: s == bad, edits=[Edit(find=bad, replace=narrowed)])
    writer = _writer(model)
    out = _verify(writer, draft, verifiers=verifiers)
    assert out == narrowed_draft
    assert not writer.blocking
    assert any("below its minimum length" in n for n in writer.notes)
