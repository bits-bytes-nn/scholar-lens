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


def test_guide_document_converts_markers_and_drops_unoffered_images() -> None:
    from scholar_lens.src.guide_pipeline import guide_document
    from scholar_lens.src.tech_guide import TechGuide

    arch = "https://docs.x.dev/img/arch.png"
    guide = TechGuide(topic="X 시작하기", body="", image_urls=[arch])
    document = (
        "## 구조\n\n아래 그림은 구조를 보여 줍니다.\n\n"
        f'[Image: alt="아키텍처", src="{arch}"]\n\n'
        "![로고](https://docs.x.dev/img/logo.svg)\n"
    )
    out = guide_document(document, guide)
    assert "[Image:" not in out
    assert f"]({arch})" in out
    assert "logo.svg" not in out
