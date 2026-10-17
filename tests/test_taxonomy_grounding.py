"""Guide flags must exist in the official docs (deterministic checks, no LLM
calls)."""

from __future__ import annotations

import pytest

from scholar_lens.src.guide_authority import official_reference, unverified_identifiers
from scholar_lens.src.web_research import (
    OfficialDocs,
    PageContent,
    ResearchCorpus,
    docs_version,
)

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
        missing = unverified_identifiers(GUIDE, OFFICIAL, ["vllm"])
        assert missing == ["--model", "--preemption-mode", "vllm_request_failure_total"]
        assert "--gpus" not in missing  # docker's own option, before "vllm"

    def test_only_requested_hosts_are_official(self) -> None:
        corpus = _corpus(
            [
                ("https://docs.vllm.ai/en/latest/a.html", OFFICIAL),
                ("https://blog.example.com/x", "--preemption-mode swap"),
            ]
        )
        official = OfficialDocs.from_urls(["https://docs.vllm.ai/en/latest/"])
        text = official_reference(official, corpus)
        assert "--preemption-mode" not in text
        assert "--preemption-mode" in unverified_identifiers(GUIDE, text, ["vllm"])

    def test_no_official_text_checks_nothing(self) -> None:
        assert unverified_identifiers(GUIDE, "", ["vllm"]) == []


def test_sources_are_labelled_by_authority() -> None:
    corpus = _corpus(
        [
            ("https://docs.vllm.ai/a", "paged attention"),
            ("https://x.io/b", "paged attention"),
        ]
    )
    text = corpus.relevant_text(
        "paged attention",
        max_chars=10_000,
        official=OfficialDocs.from_urls(["https://docs.vllm.ai/"]),
    )
    assert "## Source (official documentation): https://docs.vllm.ai/a" in text
    assert "## Source (third-party): https://x.io/b" in text
    assert "## Source: " in corpus.relevant_text("paged attention", max_chars=10_000)


@pytest.mark.parametrize(
    "url,version",
    [
        ("https://docs.vllm.ai/en/latest/quickstart.html", "latest"),
        ("https://docs.vllm.ai/en/v0.6.0/automatic_prefix_caching.html", "v0.6.0"),
        ("https://pytorch.org/docs/stable/notes/cuda.html", "stable"),
        ("https://pytorch.org/docs/2.4/notes/cuda.html", "2.4"),
        ("https://x.io/docs/0.6/guide", "0.6"),
        ("https://x.io/api/v2/users", "v2"),
        ("https://blog.x.io/2024/05/post", None),
        ("https://docs.x.io/en/guide/", None),
    ],
)
def test_docs_version(url: str, version: str | None) -> None:
    assert docs_version(url) == version


def test_official_docs_are_version_aware() -> None:
    latest = OfficialDocs.from_urls(["https://docs.vllm.ai/en/latest/quickstart/"])
    assert latest.covers("https://docs.vllm.ai/en/latest/engine_args.html")
    assert latest.covers("https://docs.vllm.ai/en/stable/engine_args.html")
    assert latest.covers("https://docs.vllm.ai/usage/metrics/")
    assert not latest.covers("https://docs.vllm.ai/en/v0.6.0/engine_args.html")
    assert not latest.covers("https://x.io/en/latest/engine_args.html")
    pinned = OfficialDocs.from_urls(["https://docs.vllm.ai/en/v0.6.0/quickstart.html"])
    assert pinned.covers("https://docs.vllm.ai/en/v0.6.0/engine_args.html")
    assert not pinned.covers("https://docs.vllm.ai/en/latest/engine_args.html")
    assert not pinned.covers("https://docs.vllm.ai/en/v0.7.0/engine_args.html")


def test_older_docs_version_is_labelled_and_not_official() -> None:
    corpus = _corpus(
        [
            ("https://docs.vllm.ai/en/latest/a", "prefix caching"),
            ("https://docs.vllm.ai/en/v0.6.0/b", "prefix caching"),
        ]
    )
    official = OfficialDocs.from_urls(["https://docs.vllm.ai/en/latest/quickstart/"])
    text = corpus.relevant_text("prefix caching", max_chars=10_000, official=official)
    assert "(official documentation): https://docs.vllm.ai/en/latest/a" in text
    assert "(older docs version): https://docs.vllm.ai/en/v0.6.0/b" in text
    reference = official_reference(official, corpus)
    assert reference.count("prefix caching") == 1


def test_underscore_flags_are_read_whole_and_match_either_spelling() -> None:
    body = (
        "```bash\naccelerate launch --multi_gpu --num_processes 2 train.py\n"
        "torchrun --nproc_per_node=8 --max-restarts=3 train.py\n```"
    )
    official = (
        "Use --multi_gpu and --num_processes. torchrun --nproc-per-node, --max_restarts"
    )
    found = unverified_identifiers(body, official, ["accelerate", "torchrun"])
    assert found == []
    assert (
        unverified_identifiers(body, "nothing here", ["accelerate"])[0] == "--multi_gpu"
    )


def test_inline_code_checks_bare_flags_not_other_tools() -> None:
    found = unverified_identifiers(
        "`--stale-flag` 와 `curl --silent`", "nothing", ["vllm"]
    )
    assert found == ["--stale-flag"]


def test_single_word_flags_are_not_excused_by_prose() -> None:
    body = "```bash\nvllm serve m --model x --gpu 1\n```"
    official = "Pick the model and the gpu count. vllm serve --port 8000"
    assert unverified_identifiers(body, official, ["vllm"]) == ["--model", "--gpu"]


def test_other_dash_style_only_counts_when_the_docs_use_both() -> None:
    body = "```bash\nruff check --unsafe_fixes\n```"
    assert unverified_identifiers(
        body, "ruff check --unsafe-fixes --fix", ["ruff"]
    ) == ["--unsafe_fixes"]


def test_underscore_flag_is_not_excused_by_a_python_argument() -> None:
    body = "```bash\nhf download repo --local_dir out\n```"
    official = "hf download repo --local-dir out\n\nhf_hub_download(local_dir=...)"
    assert unverified_identifiers(body, official, ["hf"]) == ["--local_dir"]


def test_fence_language_tag_does_not_name_the_product() -> None:
    """A "```python" fence is not a `python` command: an illustrative
    simulation's arguments are not the product's API."""
    body = (
        "```python\nalloc = BlockAllocator(num_blocks=8)\n```\n\n"
        "```bash\npython -m vllm.entrypoints.openai.api_server --stale-flag\n```"
    )
    assert unverified_identifiers(body, "nothing", ["vllm", "python"]) == [
        "--stale-flag"
    ]
