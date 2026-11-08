"""Properties of the verifier set and the edit transaction (docs/engine.md,
4.1 and 4.4)."""

from __future__ import annotations

import random

import pytest

from scholar_lens.domain.edit_transaction import rejudged_units, transact
from scholar_lens.domain.evidence import Evidence, EvidenceLedger
from scholar_lens.domain.evidence_draft import (
    apply_edits,
    free_numbers,
    lost_assets,
    prose_sentences,
    section_problems,
    structure_lost,
)
from scholar_lens.domain.post_spec import Limits
from scholar_lens.domain.review_plan import Edit
from scholar_lens.domain.review_plan import apply_edits as guarded
from scholar_lens.domain.verifiers import (
    Problem,
    VerifierSet,
    item_problems,
    messages,
    no_checks,
)

CUT = Limits().max_prose_cut
CODE = "```python\nclient = Client(retries=3)\n```"
TABLE = "| 방법 | 점수 |\n| --- | --- |\n| LoRA | 높음 |"


def _section(*blocks: str) -> str:
    return "## 사용법\n\n" + "\n\n".join(blocks) + "\n"


# ---- the table edit split in two ----------------------------------------------


@pytest.mark.parametrize("columns", [1, 2, 3])
def test_a_table_edit_split_into_header_and_delimiter_edits_applies(
    columns: int,
) -> None:
    text = _section("표를 봅니다.", TABLE, "끝입니다.")
    extra_head = "".join(f" 열{k} |" for k in range(columns))
    extra_rule = " --- |" * columns
    extra_row = " 값 |" * columns
    edits = [
        ("| 방법 | 점수 |", "| 방법 | 점수 |" + extra_head),
        ("| --- | --- |", "| --- | --- |" + extra_rule),
        ("| LoRA | 높음 |", "| LoRA | 높음 |" + extra_row),
    ]
    # Alone, the header edit leaves a header wider than its delimiter row.
    _, alone, _ = guarded(
        text, [Edit(find=edits[0][0], replace=edits[0][1])], max_edits=5
    )
    assert alone == 0
    fixed, applied, skipped = apply_edits(text, edits)
    assert applied == len(edits) and not skipped
    assert "| --- | --- |" + extra_rule in fixed


def test_a_batch_that_fails_together_keeps_the_edits_that_pass_alone() -> None:
    text = _section("첫 문장입니다.", "![구조](https://x/a.png)", "끝입니다.")
    fixed, applied, skipped = apply_edits(
        text,
        [("첫 문장입니다.", "고친 문장입니다."), ("![구조](https://x/a.png)", "")],
    )
    assert applied == 1 and "고친 문장입니다." in fixed
    assert "https://x/a.png" in fixed and any("only figure" in r for r in skipped)


# ---- a copy that remains -------------------------------------------------------


@pytest.mark.parametrize(
    "asset", [CODE, TABLE, "$$\ny = Wx\n$$", "![구조](https://x/a.png)"]
)
@pytest.mark.parametrize("copy", ["section", "rest", "nowhere"])
def test_a_block_may_go_only_when_a_copy_remains(asset: str, copy: str) -> None:
    blocks = ["설명입니다.", asset, "다음 설명입니다."]
    if copy == "section":
        blocks.append(asset)
    text = _section(*blocks)
    rest = _section("앞 절입니다.", asset) if copy == "rest" else _section("앞 절.")
    edits = [(f"{asset}\n\n다음", "다음")]
    fixed, applied, _ = apply_edits(text, edits, rest=rest)
    revision = transact(
        text, edits, verifiers=no_checks(), max_prose_cut=CUT, rest=rest
    )
    lost = structure_lost(text, fixed, rest=rest, max_prose_cut=CUT)
    if copy == "nowhere":
        assert applied == 0 and not revision.taken
    else:
        assert applied == 1 and revision.taken and not lost
        assert fixed.count(asset) == (1 if copy == "section" else 0)


@pytest.mark.parametrize(
    "asset", [CODE, TABLE, "$$\ny = Wx\n$$", "![구조](https://x/a.png)"]
)
@pytest.mark.parametrize("copy", ["earlier", "later", "both"])
def test_a_code_block_may_go_only_when_a_copy_stays_before_it(
    asset: str, copy: str
) -> None:
    """Code runs in reading order: deleting a block whose copy is only in a
    later section breaks the code between them. Figures, tables and
    equations keep the any-section rule."""
    text = _section("설명입니다.", asset, "다음 설명입니다.")
    other = _section("다른 절입니다.", asset)
    earlier = other if copy in ("earlier", "both") else _section("앞 절.")
    later = other if copy in ("later", "both") else _section("뒤 절.")
    rest = f"{earlier}\n\n{later}"
    edits = [(f"{asset}\n\n다음", "다음")]
    revision = transact(
        text, edits, verifiers=no_checks(), max_prose_cut=CUT, rest=rest, later=later
    )
    fixed, applied, _ = apply_edits(text, edits, rest=rest, later=later)
    removable = copy != "later" or asset != CODE
    assert revision.taken == removable and (applied == 1) == removable
    gone = _section("설명입니다.", "다음 설명입니다.")
    assert (lost_assets(text, gone, rest, None, later) == "") == removable
    # Without ``later``, the whole rest counts as before the section.
    assert lost_assets(text, gone, rest) == ""


def test_an_asset_fixed_in_place_is_not_lost() -> None:
    before = _section("설명입니다.", CODE)
    after = before.replace("retries=3", "retries=5")
    assert lost_assets(before, after) == ""
    assert lost_assets(before, _section("설명입니다.")) == "code block"


def test_the_prose_cut_guard_is_the_spec_limit() -> None:
    before = _section("가 " * 40 + "끝입니다.")
    after = _section("가 " * 10 + "끝입니다.")
    assert structure_lost(before, after, max_prose_cut=CUT)
    assert not structure_lost(before, after, max_prose_cut=0.9)


# ---- problem sets ----------------------------------------------------------------


def _numbers(markdown: str) -> list[Problem]:
    """A stand-in verifier: every sentence holding "N개" is a problem of
    one kind, quoting its sentence."""
    out = []
    for line in markdown.split("\n"):
        if "개" in line and any(c.isdigit() for c in line):
            out.append(Problem("numbers", f"numbers without evidence: {line}", line))
    return out


def _headings(markdown: str) -> list[Problem]:
    return [Problem("subheading", "no ### headings")] if "\n### " in markdown else []


VERIFIERS = VerifierSet("test", (_numbers, _headings))


def test_a_problem_key_drops_its_position_and_digits() -> None:
    a = Problem(
        "numbers", "numbers [12] without evidence: 첫 문장 12개", "첫 문장 12개"
    )
    b = Problem(
        "numbers", "numbers [7] without evidence: 둘째 문장 7개", "둘째 문장 7개"
    )
    assert a.key == b.key
    assert a.key != Problem("other", a.text, a.where).key


def test_a_batch_that_adds_a_new_kind_of_problem_is_rejected() -> None:
    text = _section("첫 문장입니다.", "끝입니다.")
    revision = transact(
        text,
        [("첫 문장입니다.", "첫 문장입니다.\n\n### 세부\n\n세부 설명입니다.")],
        verifiers=VERIFIERS,
        max_prose_cut=CUT,
    )
    assert not revision.taken and revision.text == text
    assert [p.kind for p in revision.added] == ["subheading"]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_a_problem_moved_is_kept_and_one_more_of_a_form_is_added(
    count: int,
) -> None:
    text = _section("모델은 3개 층을 씁니다.", "끝입니다.")
    moved = "\n\n".join(f"설정은 {k + 4}개 값을 받습니다." for k in range(count))
    revision = transact(
        text,
        [("모델은 3개 층을 씁니다.", f"모델은 층을 씁니다.\n\n{moved}")],
        verifiers=VERIFIERS,
        max_prose_cut=CUT,
    )
    # Normalised problems are compared as a multiset: the one problem
    # moved (and reworded) is the same problem; each one more is new.
    assert revision.taken is (count == 1)
    assert len(revision.added) == count - 1
    assert all(p.kind == "numbers" for p in revision.added)
    if count == 1:
        assert "설정은 4개 값을 받습니다." in rejudged_units(text, revision.text)


@pytest.mark.parametrize("before_count", [0, 1, 2])
@pytest.mark.parametrize("after_count", [0, 1, 2, 3])
def test_added_counts_the_growth_of_each_normalised_form(
    before_count: int, after_count: int
) -> None:
    def problems(n: int, place: str) -> list[Problem]:
        return [
            Problem("foreign", f"foreign evidence [E{k}] in: {place}{k}", f"{place}{k}")
            for k in range(n)
        ]

    added = VERIFIERS.added(problems(before_count, "앞 "), problems(after_count, "뒤 "))
    assert len(added) == max(0, after_count - before_count)


def test_rejudged_units_are_the_changed_ones_and_the_unit_after() -> None:
    before = _section("가는 문장입니다. 나는 문장입니다. 다는 문장입니다.", TABLE)
    after = before.replace("나는 문장입니다.", "나는 고친 문장입니다.")
    # The rewritten sentence and the one whose previous sentence changed.
    assert rejudged_units(before, after) == [
        "나는 고친 문장입니다.",
        "다는 문장입니다.",
    ]


@pytest.mark.parametrize(
    "owned, taken",
    [
        # A figure given to another section (or to none) may go: its owner
        # gets it back, and nothing else should show it.
        (frozenset(), True),
        (frozenset({"https://x/other.png"}), True),
        # The section's own only copy stays.
        (frozenset({"https://x/f.png"}), False),
        # Without ownership, every only copy is protected.
        (None, False),
    ],
)
def test_only_a_sections_own_figure_is_protected(
    owned: frozenset[str] | None, taken: bool
) -> None:
    section = (
        "## 구조\n\n캐시는 요청을 묶습니다 [[E1]].\n\n"
        "![다른 절의 그림](https://x/f.png)\n\n남은 문장입니다 [[E1]].\n"
    )
    revision = transact(
        section,
        [("![다른 절의 그림](https://x/f.png)\n\n", "")],
        verifiers=VerifierSet("구조", (), owned),
        max_prose_cut=Limits().max_prose_cut,
    )
    assert revision.taken is taken
    assert ("https://x/f.png" in revision.text) is not taken


# ---- a section that ends mid-sentence ------------------------------------------


@pytest.mark.parametrize(
    "cut",
    [
        "끝은 다음과 같",  # a Korean sentence cut off
        "The batch is then sent to the",  # an English one
        "그래서 이 설정은",
    ],
)
@pytest.mark.parametrize("where", ["replace", "append"])
def test_an_edit_that_leaves_the_section_mid_sentence_is_not_taken(
    cut: str, where: str
) -> None:
    text = _section("첫 문장입니다.", "끝입니다.")
    edit = (
        ("끝입니다.", cut) if where == "replace" else ("끝입니다.", f"끝입니다. {cut}")
    )
    revision = transact(text, [edit], verifiers=no_checks(), max_prose_cut=CUT)
    assert not revision.taken and revision.text == text
    assert revision.rejected == "it stops mid-sentence"


@pytest.mark.parametrize(
    "ending",
    [
        "끝은 이렇습니다.",
        "The batch is sent.",
        "무엇일까요?",
        "- 목록 항목",
        TABLE,
        CODE,
    ],
)
def test_an_edit_that_leaves_a_complete_ending_is_taken(ending: str) -> None:
    text = _section("첫 문장입니다.", "끝입니다.")
    revision = transact(
        text,
        [("끝입니다.", "끝입니다.\n\n" + ending)],
        verifiers=no_checks(),
        max_prose_cut=CUT,
    )
    assert revision.taken and revision.text.rstrip().endswith(ending.rstrip())


def _prose(units: list[str]) -> str:
    return "## S\n\n" + "\n\n".join(units) + "\n"


def test_rejudged_units_are_the_changes_and_the_unit_after_each() -> None:
    """Property, over generated sections and single edits: nothing is
    re-judged without a change; every changed unit is; an insertion or a
    rewrite re-judges the unit after it, a deletion the unit that moved up."""
    rng = random.Random(4242)
    for _ in range(300):
        units = [f"문장 {k}번은 짧습니다." for k in rng.sample(range(50), 6)]
        k = rng.randrange(len(units))
        new = "새로 쓴 문장입니다."
        after = list(units)
        kind = rng.choice(["insert", "rewrite", "delete"])
        if kind == "insert":
            after.insert(k, new)
        elif kind == "rewrite":
            after[k] = new
        else:
            del after[k]
        before_md, after_md = _prose(units), _prose(after)
        assert rejudged_units(before_md, before_md) == []
        got = rejudged_units(before_md, after_md)
        nxt = k + 1 if kind != "delete" else k
        expected = ([new] if kind != "delete" else []) + (
            [after[nxt]] if nxt < len(after) else []
        )
        assert got == expected, (kind, k)


def _deep_checks(limits: Limits) -> VerifierSet:
    """A deep section's shared checks, as the engine builds them."""
    ledger = EvidenceLedger()

    def shared(markdown: str) -> list[Problem]:
        return section_problems(
            markdown,
            title="사용법",
            own=[],
            earlier=[],
            ledger=ledger,
            earlier_numbers={},
            figure_urls=[],
            free_numbers=free_numbers(),
            depth="deep",
            limits=limits,
        )

    return VerifierSet("사용법", (shared,))


OTHER = "| 모델 | 속도 |\n| --- | --- |\n| 작은 것 | 빠름 |"


@pytest.mark.parametrize("copy", ["rest", "nowhere"])
def test_deleting_a_duplicate_table_is_not_held_to_the_visuals_minimum(
    copy: str,
) -> None:
    """A deep section's repeated table goes when another section keeps it,
    even though the section falls below its visuals minimum; the only copy
    of a table stays."""
    text = _section("표를 봅니다.", TABLE, "다음 표입니다.", OTHER, "끝입니다.")
    rest = _section("앞 절입니다.", OTHER) if copy == "rest" else _section("앞 절.")
    revision = transact(
        text,
        [(f"\n\n{OTHER}", "")],
        verifiers=_deep_checks(Limits()),
        max_prose_cut=CUT,
        rest=rest,
    )
    assert revision.taken is (copy == "rest")
    if copy == "rest":
        assert OTHER not in revision.text and TABLE in revision.text


FIGURE = "![구조](https://x/a.png)"


@pytest.mark.parametrize(
    ("blocks", "edits", "rest"),
    [
        # A table repeated within the section only: the copy left is the
        # section's own, nothing another section keeps.
        (
            ["표를 봅니다.", TABLE, "다음 표입니다.", TABLE, "끝입니다."],
            [(f"{TABLE}\n\n끝입니다.", "끝입니다.")],
            _section("앞 절."),
        ),
        # A figure not given to the section (it may go) that no other
        # section shows.
        (
            ["그림을 봅니다.", FIGURE, "표입니다.", TABLE, "끝입니다."],
            [(f"\n\n{FIGURE}", "")],
            _section("앞 절."),
        ),
        # Two visuals removed, only one with a copy elsewhere.
        (
            ["표를 봅니다.", TABLE, "다음 표입니다.", OTHER, TABLE, "끝입니다."],
            [(f"\n\n{OTHER}\n\n{TABLE}", "")],
            _section("앞 절입니다.", OTHER),
        ),
    ],
    ids=["copy-in-section", "unowned-figure", "some-copied"],
)
def test_a_removal_without_copies_elsewhere_is_held_to_the_visuals_minimum(
    blocks: list[str], edits: list[tuple[str, str]], rest: str
) -> None:
    text = _section(*blocks)
    base = _deep_checks(Limits())
    # The section owns no figure: a figure may go, so only the visuals
    # minimum can hold it.
    verifiers = VerifierSet(base.name, base.checks, frozenset())
    assert not any(p.kind == "visuals" for p in verifiers.run(text))
    revision = transact(text, edits, verifiers=verifiers, max_prose_cut=CUT, rest=rest)
    assert revision.applied and not revision.taken
    assert [p.kind for p in revision.added] == ["visuals"]


def test_removing_a_duplicate_does_not_excuse_other_new_problems() -> None:
    text = _section("표를 봅니다.", TABLE, "다음 표입니다.", OTHER, "끝입니다.")
    revision = transact(
        text,
        [(f"\n\n{OTHER}", ""), ("끝입니다.", "끝입니다 [[E7]].")],
        verifiers=_deep_checks(Limits()),
        max_prose_cut=CUT,
        rest=_section("앞 절입니다.", OTHER),
    )
    assert not revision.taken and "unknown evidence" in revision.rejected


@pytest.mark.parametrize("share", [1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("need", [0, 1, 2, 3])
def test_the_length_share_and_visuals_minimum_come_from_the_limits(
    share: float, need: int
) -> None:
    text = _section("가" * 100 + "입니다.", TABLE)
    limits = Limits(max_length_share=share, min_visuals=(("deep", need),))
    problems = section_problems(
        text,
        title="사용법",
        own=[],
        earlier=[],
        ledger=EvidenceLedger(),
        earlier_numbers={},
        free_numbers=free_numbers(),
        figure_urls=[],
        target_chars=50,
        depth="deep",
        limits=limits,
    )
    kinds = {p.kind for p in problems}
    length = len(" ".join(prose_sentences(text)))
    assert ("too_long" in kinds) is (length > share * 50)
    assert ("visuals" in kinds) is (need > 1)


def test_the_default_limits_keep_the_former_thresholds() -> None:
    limits = Limits()
    assert limits.max_length_share == 1.5
    assert dict(limits.min_visuals) == {"deep": 2, "standard": 1, "brief": 0}


# ---- the program check as transaction problems ----------------------------------


def _program_set(later: str) -> VerifierSet:
    from scholar_lens.domain.program import program_problems

    def whole(markdown: str) -> list[Problem]:
        return program_problems([("앞", markdown, True), ("뒤", later, True)])

    return VerifierSet("앞", (whole,))


def test_a_block_rewritten_with_the_same_fault_adds_no_program_problem() -> None:
    before = _section("```python\n@serve.batch(\n```")
    after = _section("```python\nfrom ray import serve\n\n@serve.batch(\n```")
    assert _program_set("").added(before, after) == []


def test_one_more_undefined_name_is_a_new_program_problem() -> None:
    later = _section("```python\nclient.run(store)\n```")
    before = _section("```python\nclient, store = 1, 2\n```")
    after = _section("```python\nstore = 2\n```")
    (added,) = _program_set(later).added(before, after)
    assert added.kind == "program" and added.text.endswith("client")
    # The later block already lacking a name: one more is still new.
    fewer = _section("```python\nclient = 1\n```")
    assert _program_set(later).added(fewer, _section("```python\nx = 1\n```"))


# ---- the minimum length ---------------------------------------------------------

SENTENCE = "가나다라마바사아자차 카타파하입니다."


def _length_problems(
    markdown: str, target: int, *, minimum: bool = True
) -> list[Problem]:
    return section_problems(
        markdown,
        title="사용법",
        own=[],
        earlier=[],
        ledger=EvidenceLedger(),
        earlier_numbers={},
        free_numbers=free_numbers(),
        figure_urls=[],
        target_chars=target,
        limits=Limits(),
        minimum_length=minimum,
    )


def _short(problems: list[Problem]) -> bool:
    return any(p.kind == "too_short" for p in problems)


@pytest.mark.parametrize("sentences", [1, 5, 10, 20, 40])
@pytest.mark.parametrize("target", [400, 799, 800, 1_000, 2_500])
def test_a_section_below_its_share_of_the_target_is_too_short(
    sentences: int, target: int
) -> None:
    limits = Limits()
    text = _section(" ".join([SENTENCE] * sentences))
    length = len(" ".join(prose_sentences(text)))
    expected = (
        target >= limits.min_length_floor and length < limits.min_length_share * target
    )
    assert _short(_length_problems(text, target)) == expected
    # The TL;DR has no minimum.
    assert not _short(_length_problems(text, target, minimum=False))


def test_a_cutting_fix_that_leaves_the_section_short_is_not_taken() -> None:
    target = 1_000
    sentences = [f"{SENTENCE[:-1]} {k}번입니다." for k in range(30)]
    text = _section(" ".join(sentences))
    verifiers = VerifierSet("사용법", (lambda md: _length_problems(md, target),))
    assert not _short(verifiers.run(text))
    cut = [(" ".join(sentences[3:]), "")]
    revision = transact(text, cut, verifiers=verifiers, max_prose_cut=1.0)
    assert not revision.taken and "too_short" in {p.kind for p in revision.added}
    # A deletion of unsupported content tolerates it (the caller notes it).
    tolerated = transact(
        text, cut, verifiers=verifiers, max_prose_cut=1.0, tolerated=("too_short",)
    )
    assert tolerated.taken


@pytest.mark.parametrize(
    ("blocks", "count"),
    [
        (["앞입니다.", "---", "뒤입니다."], 0),
        (["앞입니다.", "***", "뒤입니다."], 0),
        (["앞입니다.", TABLE, "---", "뒤입니다."], 1),
        (["앞입니다.", TABLE, OTHER, FIGURE, "$$\ny = Wx\n$$"], 4),
    ],
)
def test_a_rule_is_not_a_table_of_the_visuals_count(
    blocks: list[str], count: int
) -> None:
    from scholar_lens.domain.evidence_draft import visual_count

    assert visual_count(_section(*blocks)) == count


@pytest.mark.parametrize("cap", [1, 2, 5])
def test_the_edits_one_fix_may_apply_are_the_limit(cap: int) -> None:
    sentences = [f"문장 {k}입니다." for k in range(4)]
    text = _section(*sentences)
    edits = [(s, s.replace("입니다", "이었습니다")) for s in sentences]
    revision = transact(
        text, edits, verifiers=no_checks(), max_prose_cut=CUT, max_edits=cap
    )
    assert revision.applied == min(cap, len(edits))
    assert any("over the cap" in r for r in revision.skipped) is (cap < len(edits))


def _section_set(target: int = 0) -> VerifierSet:
    from scholar_lens.domain.evidence_draft import section_problems

    def check(markdown: str) -> list[Problem]:
        return section_problems(
            markdown,
            title="구조",
            own=(),
            earlier=(),
            ledger=EvidenceLedger(),
            earlier_numbers={},
            figure_urls=(),
            free_numbers="",
            target_chars=target,
        )

    return VerifierSet("구조", (check,))


def test_an_edit_near_an_existing_rule_slip_adds_no_problem() -> None:
    from scholar_lens.domain.quality_gate import rule_issues

    section = (
        "## 구조\n\n이 섹션에서는 스케줄러가 요청을 묶어 처리하는 과정을 "
        "설명합니다.\n\n큐는 요청을 차례로 받습니다.\n"
    )
    assert rule_issues(section), "the fixture must hold a rule slip"
    revision = transact(
        section,
        [("요청을 묶어 처리하는", "요청을 모아 처리하는")],
        verifiers=_section_set(),
        max_prose_cut=Limits().max_prose_cut,
    )
    # The slip is as it was; only its quoted surroundings changed.
    assert revision.taken, revision.rejected


@pytest.mark.parametrize("tolerated", [(), ("too_short",)])
def test_a_section_below_its_minimum_is_not_cut_further(
    tolerated: tuple[str, ...],
) -> None:
    sentences = [f"큐는 {k}번째 요청을 차례로 받아 처리합니다." for k in range(6)]
    section = "## 구조\n\n" + " ".join(sentences) + "\n"
    limits = Limits()
    # A target the section is already short of.
    target = max(
        limits.min_length_floor, int(len(section) / limits.min_length_share) * 2
    )
    assert any(p.kind == "too_short" for p in _section_set(target).run(section))
    revision = transact(
        section,
        [(sentences[-1], "")],
        verifiers=_section_set(target),
        max_prose_cut=limits.max_prose_cut,
        tolerated=tolerated,
    )
    # Only a deletion of unsupported content (which tolerates it) may cut.
    assert revision.taken is bool(tolerated)


# ---- problems that list several items ---------------------------------------

_NUMBERS = ("517", "86.4", "733", "1.25", "9410", "348")
_PAPERS = ("2305.18290", "2401.00001", "2402.11223", "2310.04444", "2209.07858")
_IMAGES = tuple(f"https://example.org/f{k}.png" for k in range(6))


def _listing_ledger() -> EvidenceLedger:
    return EvidenceLedger(
        items=[
            Evidence(
                id=f"E{k + 1}",
                paper_id=pid,
                kind="method",
                claim="방법",
                quote="a method",
                anchor="S1",
            )
            for k, pid in enumerate(_PAPERS)
        ]
    )


def _listing_verifiers(
    papers: tuple[str, ...], figures: tuple[str, ...]
) -> VerifierSet:
    ledger = _listing_ledger()

    def shared(markdown: str) -> list[Problem]:
        return section_problems(
            markdown,
            title="사용법",
            own=[e.id for e in ledger],
            earlier=[],
            ledger=ledger,
            earlier_numbers={},
            figure_urls=figures,
            free_numbers=free_numbers(),
            papers=papers,
        )

    flags = ("--alpha", "--beta", "--gamma", "--delta")

    def unknown_flags(markdown: str) -> list[Problem]:
        # A check in the shape of the guide's flags check.
        return item_problems(
            "unknown_flags",
            [f for f in flags if f in markdown],
            "flags no official page mentions: ",
            "",
        )

    return VerifierSet("사용법", (shared, unknown_flags), frozenset(figures))


def _listing_section(
    numbers: list[str], cited: list[str], images: list[str], flags: list[str]
) -> str:
    blocks = []
    if numbers:
        blocks.append("측정값은 " + " 그리고 ".join(numbers) + " 입니다.")
    blocks += [f"이 방법을 씁니다 [[E{_PAPERS.index(p) + 1}]]." for p in cited]
    blocks += [f"![그림]({u})" for u in images]
    if flags:
        blocks.append("```bash\nserve " + " ".join(flags) + "\n```")
    blocks.append("끝입니다.")
    return _section(*blocks)


def _subset(rng: random.Random, items: list[str]) -> list[str]:
    return [i for i in items if rng.random() < 0.5]


@pytest.mark.parametrize("seed", range(40))
def test_settling_some_listed_items_adds_no_problem_and_one_more_item_does(
    seed: int,
) -> None:
    rng = random.Random(seed)
    papers = tuple(rng.sample(_PAPERS, rng.randint(1, len(_PAPERS))))
    figures = tuple(rng.sample(_IMAGES, rng.randint(0, 3)))
    verifiers = _listing_verifiers(papers, figures)
    others = [u for u in _IMAGES if u not in figures]
    # A section with N items wrong of each listing kind.
    numbers = rng.sample(_NUMBERS, rng.randint(1, len(_NUMBERS) - 1))
    cited = _subset(rng, list(papers))
    images = [*_subset(rng, list(figures)), *rng.sample(others, rng.randint(0, 2))]
    flags = rng.sample(("--alpha", "--beta", "--gamma"), rng.randint(0, 3))
    before = _listing_section(numbers, cited, images, flags)
    old = verifiers.run(before)

    # Any subset of the items fixed: no problem is added.
    fixed = _listing_section(
        [n for n in numbers if rng.random() < 0.5],
        [*cited, *_subset(rng, [p for p in papers if p not in cited])],
        [
            *[u for u in images if u in figures or rng.random() < 0.5],
            *_subset(rng, [u for u in figures if u not in images]),
        ],
        [f for f in flags if rng.random() < 0.5],
    )
    assert verifiers.added(old, fixed) == []

    # One more item of any kind: exactly one problem of its kind is added.
    more: dict[str, str] = {
        "numbers_uncited": _listing_section(
            [*numbers, next(n for n in _NUMBERS if n not in numbers)],
            cited,
            images,
            flags,
        ),
        "unknown_flags": _listing_section(numbers, cited, images, [*flags, "--delta"]),
    }
    if cited:
        more["papers_uncovered"] = _listing_section(numbers, cited[1:], images, flags)
    if spare := [u for u in others if u not in images]:
        more["extra_images"] = _listing_section(
            numbers, cited, [*images, spare[0]], flags
        )
    if shown := [u for u in images if u in figures]:
        more["missing_figures"] = _listing_section(
            numbers, cited, [u for u in images if u != shown[0]], flags
        )
    for kind, after in more.items():
        assert [p.kind for p in verifiers.added(old, after)] == [kind], kind


def test_listed_items_of_one_sentence_are_shown_to_the_writer_together() -> None:
    verifiers = _listing_verifiers(_PAPERS[:3], ())
    problems = verifiers.run(_listing_section(["517", "733"], [], [], []))
    assert [p.item for p in problems if p.kind == "numbers_uncited"] == ["517", "733"]
    shown = messages(problems)
    assert sum("517" in m for m in shown) == 1
    assert any(m.startswith("numbers [517, 733] without evidence") for m in shown)
    assert any(m.startswith(f"papers [{', '.join(_PAPERS[:3])}] belong") for m in shown)
