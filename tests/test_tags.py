"""Deterministic tag normalization against the blog's post conventions."""

from __future__ import annotations

import pytest

from scholar_lens.src.post_conventions import (
    GENERATED_SPECIFIC_TAGS,
    load_local_conventions,
)
from scholar_lens.src.tags import format_tag, normalize_tags, unsupported_tags

SPEC = load_local_conventions()
WORDS = SPEC.tags.lowercase_words


class TestFormatTag:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Pan-&-Scan Image Processing", "Pan-and-Scan-Image-Processing"),
            ("Public-Dataset-Only Training", "Public-Dataset-Only-Training"),
            ("Leiden-Algorithm-For-Text", "Leiden-Algorithm-for-Text"),
            (
                "Training-&amp;-Inference-Optimization",
                "Training-and-Inference-Optimization",
            ),
            ("Block-wise-k-bit-Quantization", "Block-wise-k-bit-Quantization"),
            ("Graph/RAG", "GraphRAG"),
            ("multi_token  prediction", "Multi-token-prediction"),
            ("On-Policy Distillation", "On-Policy-Distillation"),
        ],
    )
    def test_spelling(self, raw: str, expected: str) -> None:
        assert format_tag(raw, WORDS) == expected

    def test_yaml_on_is_a_word(self) -> None:
        assert "on" in WORDS


class TestNormalizeTags:
    def test_aliases_dedup_and_controlled_split(self) -> None:
        result = normalize_tags(
            ["Sparse Mixture of Experts", "Graph-RAG", "Quantization", "graphrag"],
            ["Llama", "Not-A-Controlled-Tag"],
            title="Some Paper",
            spec=SPEC,
        )
        assert result.specific == ["GraphRAG"]
        assert result.controlled == ["Llama", "Mixture-of-Experts", "Quantization"]
        assert result.tags == [
            "GraphRAG",
            "Llama",
            "Mixture-of-Experts",
            "Quantization",
        ]
        assert result.replaced["Sparse-Mixture-of-Experts"] == "Mixture-of-Experts"
        assert "Not-a-Controlled-Tag" in result.dropped

    def test_title_restatements_removed(self) -> None:
        title = "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism"
        result = normalize_tags(
            [
                "Scaling Open-Source Language Models with Longtermism",
                "DeepSeek LLM",
                "Multi-Step Learning Rate Scheduler",
            ],
            ["DeepSeek"],
            title=title,
            spec=SPEC,
        )
        assert result.specific == ["Multi-Step-Learning-Rate-Scheduler"]
        assert result.controlled == ["DeepSeek"]

    def test_caps(self) -> None:
        result = normalize_tags(
            [f"Specific-Tag-{i}" for i in range(9)],
            SPEC.tags.controlled,
            title="T",
            spec=SPEC,
        )
        assert len(result.specific) == GENERATED_SPECIFIC_TAGS
        assert len(result.controlled) == SPEC.tags.max_controlled

    def test_no_controlled_tag_left_empty(self) -> None:
        # The validator turns this into an error; normalization does not invent one.
        assert normalize_tags(["Foo"], [], title="T", spec=SPEC).controlled == []


class TestUnsupportedTags:
    def test_phrase_or_acronym_counts(self) -> None:
        text = "We train with DPO and multi-token prediction."
        assert unsupported_tags(
            [
                "Direct-Preference-Optimization",
                "Multi-Token-Prediction",
                "Paged-Optimizers",
            ],
            text,
        ) == ["Paged-Optimizers"]


class TestFetchBlogTags:
    def test_counts_tags_across_posts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.src import post_conventions

        posts = {
            "u1": '---\ntitle: "A"\ntags: ["GraphRAG", "Knowledge-Graph"]\n---\nbody',
            "u2": '---\ntitle: "B"\ntags: ["GraphRAG"]\n---\nbody',
        }

        class Response:
            def __init__(self, payload=None, text=""):  # type: ignore[no-untyped-def]
                self._payload, self.text, self.status_code = payload, text, 200

            def raise_for_status(self) -> None:
                pass

            def json(self):  # type: ignore[no-untyped-def]
                return self._payload

        def fake_get(url, **kwargs):  # type: ignore[no-untyped-def]
            if url.endswith("/contents/_posts"):
                return Response(
                    [
                        {"type": "file", "name": "a.md", "url": "u1"},
                        {"type": "file", "name": "b.md", "url": "u2"},
                        {"type": "dir", "name": "drafts", "url": "x"},
                    ]
                )
            return Response(text=posts[url])

        monkeypatch.setattr(post_conventions.requests, "get", fake_get)
        assert post_conventions.fetch_blog_tags("o/r", "main", None) == [
            "GraphRAG",
            "Knowledge-Graph",
        ]


class TestSpellingAndStylizedNames:
    def test_reuses_blog_spelling_and_keeps_stylized_case(self) -> None:
        result = normalize_tags(
            ["Rotary position embedding", "vLLM", "Key–Value Cache"],
            ["Efficient-Attention"],
            title="T",
            spec=SPEC,
            known_tags=["Rotary-Position-Embedding"],
        )
        assert result.specific == [
            "Rotary-Position-Embedding",
            "vLLM",
            "Key-Value-Cache",
        ]
