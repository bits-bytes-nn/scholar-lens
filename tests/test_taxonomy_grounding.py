"""Controlled tags must be the source's topic; guide flags must exist in the
official docs (deterministic checks, no LLM calls)."""

from __future__ import annotations

from scholar_lens.src.formatting import build_post_meta
from scholar_lens.src.post_conventions import load_local_conventions
from scholar_lens.src.tags import off_topic_tags
from scholar_lens.src.tech_guide import official_reference, unverified_identifiers
from scholar_lens.src.web_research import PageContent, ResearchCorpus

PREFERENCE_PAPER = (
    "We fine-tune language models from human preferences. Alignment of the "
    "policy uses a Bradley-Terry model; reasoning about the reward is implicit."
)


class TestOffTopicTags:
    def test_barely_mentioned_tag_is_off_topic(self) -> None:
        assert off_topic_tags(["Alignment", "Reasoning-Models"], PREFERENCE_PAPER) == [
            "Reasoning-Models"
        ]

    def test_broad_tag_whose_words_recur_is_kept(self) -> None:
        text = "sliding window attention is efficient. " * 3
        assert off_topic_tags(["Efficient-Attention"], text) == []

    def test_acronym_counts_as_support(self) -> None:
        assert off_topic_tags(["Retrieval-Augmented-Generation"], "We use RAG.") == []


class TestControlledTagDrop:
    def _meta(self, controlled: list[str], text: str):  # type: ignore[no-untyped-def]
        return build_post_meta(
            title="A Paper",
            category_type="Paper Reviews",
            proposed_topic="",
            specific_tags=["Preference-Learning"],
            controlled_tags=controlled,
            proposed_seo_title=None,
            spec=load_local_conventions(),
            source_text=text,
        )

    def test_off_topic_controlled_tag_dropped_with_a_note(self) -> None:
        meta = self._meta(["Alignment", "Reasoning-Models"], PREFERENCE_PAPER)
        assert "Reasoning-Models" not in meta.tags and "Alignment" in meta.tags
        assert any("Reasoning-Models" in note for note in meta.notes)

    def test_last_controlled_tag_is_kept(self) -> None:
        meta = self._meta(["Reasoning-Models"], PREFERENCE_PAPER)
        assert "Reasoning-Models" in meta.tags and meta.notes == []


GUIDE = """## 서빙

```bash
docker run --gpus all vllm/vllm-openai:latest --model Qwen \\
  --max-model-len 8192 --preemption-mode swap
```

지표 `vllm:num_requests_running`과 `vllm_request_failure_total`을 봅니다. 서버는 `--port`로 바꿉니다.
"""
OFFICIAL = "Use --max-model-len to cap the context and --port to bind. Metric vllm:num_requests_running."


def _corpus(pages: list[tuple[str, str]]) -> ResearchCorpus:
    return ResearchCorpus(
        pages=[PageContent(url=u, title="t", text=t) for u, t in pages]
    )


class TestUnverifiedIdentifiers:
    def test_flags_and_metrics_missing_from_official_pages(self) -> None:
        missing = unverified_identifiers(GUIDE, OFFICIAL, "vllm")
        assert missing == ["--model", "--preemption-mode", "vllm_request_failure_total"]
        assert "--gpus" not in missing  # docker's own option, before "vllm"

    def test_only_requested_hosts_are_official(self) -> None:
        corpus = _corpus(
            [
                ("https://docs.vllm.ai/en/latest/a.html", OFFICIAL),
                ("https://blog.example.com/x", "--preemption-mode swap"),
            ]
        )
        text, product = official_reference(["https://docs.vllm.ai/en/latest/"], corpus)
        assert product == "vllm" and "--preemption-mode" not in text
        assert "--preemption-mode" in unverified_identifiers(GUIDE, text, product)

    def test_no_official_text_checks_nothing(self) -> None:
        assert unverified_identifiers(GUIDE, "", "vllm") == []


def test_sources_are_labelled_by_authority() -> None:
    corpus = _corpus(
        [
            ("https://docs.vllm.ai/a", "paged attention"),
            ("https://x.io/b", "paged attention"),
        ]
    )
    text = corpus.relevant_text(
        "paged attention", max_chars=10_000, official_hosts={"docs.vllm.ai"}
    )
    assert "## Source (official documentation): https://docs.vllm.ai/a" in text
    assert "## Source (third-party): https://x.io/b" in text
    assert "## Source: " in corpus.relevant_text("paged attention", max_chars=10_000)
