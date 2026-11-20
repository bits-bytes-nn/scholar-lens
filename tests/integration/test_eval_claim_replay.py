"""Scoring of the claim-check replay (no model calls)."""

from __future__ import annotations

from eval.claim_replay import load_truth, report, score_case

CASE = {
    "truth": [
        {"id": "swap", "anchor": "정확도가 높았고(64.8 대 70.2)"},
        {"id": "cause", "anchor": "검색 자체가 나아졌습니다"},
    ]
}


def test_a_line_finds_an_entry_only_by_the_sentence_it_copies() -> None:
    problems = [
        # The disputed sentence holds the anchor, with markup and spacing.
        '- "SOTUQA에서는 ReWOO가 ReAct보다 **정확도가  높았고**(64.8 대 70.2)" || '
        "wrong: Table 2 gives 70.2 for ReWOO",
        # The anchor only in the reason does not count.
        "- 다른 문장 || unsupported: 검색 자체가 나아졌습니다 is not in the paper",
    ]
    score = score_case("rewoo", CASE, problems)
    assert score.found == ["swap"]
    assert score.missed == ["cause"]
    assert score.other == [problems[1]]


def test_flags_on_a_clean_baseline_are_counted_apart() -> None:
    clean = score_case("lora", {"truth": [], "clean": True}, ["- a || wrong: b"])
    other = score_case("x", {"truth": [], "clean": False}, ["- c || unsupported: d"])
    summary = report([clean, other])
    assert summary["recall"] == "0/0"
    assert summary["clean_flags"] == 1
    assert summary["unconfirmed_flags"] == 1


def test_the_truth_set_loads_with_an_anchor_per_entry() -> None:
    cases = load_truth()
    entries = [e for c in cases.values() for e in c.get("truth") or []]
    assert len(entries) == 12
    assert all(e["anchor"].strip() and e["source"] for e in entries)
    assert {c["kind"] for c in cases.values()} == {"review", "summary"}
    assert any(c.get("clean") for c in cases.values())


def test_a_review_is_checked_without_its_tldr_and_closing() -> None:
    from eval.claim_replay import checked_text

    body = (
        "## TL;DR\n- headline\n\n- - -\n\n## 방법\n본문 문장.\n\n"
        "## 결과\n결과 문장.\n\n## 한계와 시사점\n마무리 문장.\n\n- - -\n### References\n* r"
    )
    review = checked_text("review", body)
    assert review.startswith("## 방법") and "결과 문장." in review
    assert "headline" not in review and "마무리 문장." not in review
    summary = checked_text("summary", body)
    assert "headline" in summary and "마무리 문장." in summary
    assert "References" not in summary


def test_a_final_check_entry_is_listed_but_not_scored() -> None:
    case = {
        "truth": [
            {"id": "tldr", "anchor": "최대 9.2배", "check": "final"},
            {"id": "body", "anchor": "검색 자체가 나아졌습니다"},
        ]
    }
    # The TL;DR entry is not scored even when a line quotes it.
    score = score_case(
        "gepa", case, ["- 프롬프트는 최대 9.2배 짧았습니다. || wrong: x"]
    )
    assert score.missed == ["body"] and score.found == []
    # A line split at an ellipsis still finds the sentence.
    split = ["- 부트스트래핑으로 검색 자체가 ... 나아졌습니다. || unsupported: x"]
    assert score_case("gepa", case, split).found == ["body"]
