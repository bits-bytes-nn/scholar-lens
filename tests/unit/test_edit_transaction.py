"""Properties of the verifier set and the edit transaction (docs/engine.md,
4.1 and 4.4)."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from scholar_lens.domain.edit_transaction import changed_units, rejudged_units, transact
from scholar_lens.domain.evidence import EvidenceLedger
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
from scholar_lens.domain.verifiers import Problem, VerifierSet, no_checks

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
        assert "설정은 4개 값을 받습니다." in revision.changed


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


def test_changed_units_are_the_new_and_changed_sentences_only() -> None:
    before = _section("가는 문장입니다. 나는 문장입니다.", TABLE)
    after = before.replace("나는 문장입니다.", "나는 고친 문장입니다.").replace(
        "| LoRA | 높음 |", "| LoRA | 낮음 |"
    )
    assert changed_units(before, after) == ["나는 고친 문장입니다.", "| LoRA | 낮음 |"]


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
        assert not Counter(changed_units(before_md, after_md)) - Counter(got)
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
