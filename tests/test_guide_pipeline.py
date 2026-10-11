"""Deterministic helpers of the guide pipeline (no LLM calls)."""

from __future__ import annotations

from scholar_lens.src.guide_pipeline import guide_branch_id


def test_branch_id_follows_the_source_urls_not_the_title() -> None:
    a = guide_branch_id(["https://docs.vllm.ai/en/latest/", "https://X.org/a"])
    b = guide_branch_id(["https://x.org/a/", "https://docs.vllm.ai/en/latest"])
    assert a == b and a.startswith("guide-") and len(a) == len("guide-") + 12
    assert guide_branch_id(["https://x.org/b"]) != a


def test_branch_id_tells_pages_apart_by_query() -> None:
    intro = guide_branch_id(["https://docs.example.com/view?id=intro"])
    advanced = guide_branch_id(["https://docs.example.com/view?id=advanced"])
    assert intro != advanced
    assert intro == guide_branch_id(["https://docs.example.com/view?id=intro#top"])
