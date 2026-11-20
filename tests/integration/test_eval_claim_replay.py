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
