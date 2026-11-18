"""Tests for settling a checker's quoted finding after an edit."""

from __future__ import annotations

import pytest

from scholar_lens.domain.quoted import claim_fixed, finding_fixed, quoted_passages

DRAFT = (
    "하지만 보상 계산이 느리면 강화학습은 표본 효율이 떨어집니다. "
    '| 4 | 94.44 | 출력이 "Reasoning"과 "LLM Request"로 구조화됩니다 |'
)


@pytest.mark.parametrize(
    "finding,edited,fixed",
    [
        (  # the quoted sentence was rewritten
            '"하지만 보상 계산이 느리면 강화학습은 표본 효율이 떨어집니다." → '
            '원문은 "RL can be sample-inefficient"로 가능성만 말합니다.',
            "보상 계산이 느리면 강화학습은 표본 효율이 떨어질 수 있습니다.",
            True,
        ),
        (  # still there, wrapped differently
            '"하지만 보상 계산이 느리면 강화학습은 표본 효율이 떨어집니다." → 완화',
            "하지만 보상 계산이 느리면\n강화학습은 표본 효율이 떨어집니다.",
            False,
        ),
        (  # an escaped quote inside the span, and an ellipsis
            '"| 4 | 94.44 | 출력이 \\"Reasoning\\"과 ..." — 노드 2 지시문에 이미 있습니다',
            "| 4 | 94.44 | 구조가 바뀝니다 |",
            True,
        ),
        (  # quotes only the paper: nothing of the draft to look for
            '원문은 "RL can be sample-inefficient"로 가능성만 말합니다.',
            "",
            False,
        ),
        ("근거를 보강합니다.", "", False),  # quotes nothing
    ],
)
def test_finding_fixed(finding: str, edited: str, fixed: bool) -> None:
    assert finding_fixed(finding, DRAFT, edited) is fixed


def test_a_quote_of_the_paper_is_not_a_passage_of_the_draft() -> None:
    finding = '"하지만 보상 계산이" → 원문은 "RL can be sample-inefficient"입니다.'
    assert quoted_passages(finding, DRAFT) == [["하지만 보상 계산이"]]


def test_a_finding_quoting_two_passages_stays_open_while_one_remains() -> None:
    finding = '"보상 계산이 느리면" 및 "LLM Request" 둘 다 고칩니다'
    assert not finding_fixed(finding, DRAFT, "LLM Request 칸은 그대로")
    assert finding_fixed(finding, DRAFT, "둘 다 고쳤습니다")


@pytest.mark.parametrize(
    "problem,edited,fixed",
    [
        (
            "- LoRA는 모든 기준선을 이깁니다. || wrong: 일부만",
            "LoRA는 일부에서 앞섭니다.",
            True,
        ),
        (
            "- LoRA는 모든 기준선을 이깁니다. || wrong: 일부만",
            "LoRA는 모든 기준선을 이깁니다.",
            False,
        ),
        ("- 원문에 없는 문장입니다. || wrong: x", "", False),  # not from the old text
    ],
)
def test_claim_fixed(problem: str, edited: str, fixed: bool) -> None:
    assert claim_fixed(problem, "LoRA는 모든 기준선을 이깁니다.", edited) is fixed


def test_markup_added_around_a_passage_does_not_fix_it() -> None:
    finding = '"하지만 보상 계산이 느리면 강화학습은 표본 효율이 떨어집니다." → 완화'
    edited = "하지만 **보상 계산**이 느리면 [강화학습](https://x.org)은 표본 효율이 떨어집니다."
    assert not finding_fixed(finding, DRAFT, edited)


def test_italics_or_math_marks_added_around_a_passage_do_not_fix_it() -> None:
    finding = '"하지만 보상 계산이 느리면 강화학습은 표본 효율이 떨어집니다." → 완화'
    edited = "하지만 *보상 계산*이 느리면 강화학습은 표본 효율이 떨어집니다."
    assert not finding_fixed(finding, DRAFT, edited)


def test_a_fix_that_adds_a_wrong_claim_is_not_kept_even_with_fewer_problems() -> None:
    from scholar_lens.domain.quoted import fix_kept

    before = ["- A는 B입니다. || unsupported: x", "- C는 D입니다. || unsupported: y"]
    after = ["- E는 F입니다. || wrong: z"]
    assert not fix_kept(
        before, after, {"t": "A는 B입니다. C는 D입니다."}, {"t": "E는 F입니다."}
    )
    assert fix_kept(
        before, before[:1], {"t": "A는 B입니다. C는 D입니다."}, {"t": "A는 B입니다."}
    )


def test_a_kept_fix_still_notes_first_findings_left_in_the_text() -> None:
    from scholar_lens.domain.quoted import kept_problems

    before = [
        "- A는 B입니다. || wrong: x",
        "- C는 D입니다. || unsupported: y",
        "- E는 F입니다. || unsupported: z",
    ]
    after = ["- C는 D입니다. || unsupported: y again"]
    new = {"t": "A는 고쳤습니다. C는 D입니다. E는 F입니다."}
    kept = kept_problems(before, after, new)
    # A was fixed; C was found again (once); E was missed by the recheck.
    assert kept == [after[0], before[2]]


def test_a_bolded_wrong_label_counts() -> None:
    from scholar_lens.domain.quoted import fix_kept

    before = ["- A는 B입니다. || unsupported: x"]
    after = ["- C는 D입니다. || **wrong**: y"]
    assert not fix_kept(before, after, {"t": "A는 B입니다."}, {"t": "C는 D입니다."})


def test_a_finding_reported_again_with_other_punctuation_is_noted_once() -> None:
    from scholar_lens.domain.quoted import kept_problems

    before = ["- C는 D입니다. || unsupported: y"]
    after = ["- C는 D입니다 || unsupported: y"]  # no final period
    assert kept_problems(before, after, {"t": "C는 D입니다."}) == after


def test_a_fix_whose_recheck_misses_old_findings_is_not_kept_as_better() -> None:
    from scholar_lens.domain.quoted import fix_kept

    before = [f"- {s}는 그대로입니다. || unsupported: x" for s in ("A", "B", "C")]
    after = ["- G는 새로 썼습니다. || unsupported: y"]
    new = {
        "t": "A는 그대로입니다. B는 그대로입니다. C는 그대로입니다. G는 새로 썼습니다."
    }
    assert not fix_kept(before, after, {"t": new["t"]}, new)


def test_an_underscored_wrong_label_counts() -> None:
    from scholar_lens.domain.quoted import fix_kept

    before = ["- A는 B입니다. || unsupported: x"]
    after = ["- C는 D입니다. || _wrong_: y"]
    assert not fix_kept(before, after, {"t": "A는 B입니다."}, {"t": "C는 D입니다."})
