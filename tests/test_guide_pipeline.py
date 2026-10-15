"""The guide pipeline's deterministic parts (no LLM calls)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
import pytest

from scholar_lens.configs import Github
from scholar_lens.configs.config import Config
from scholar_lens.src import guide_pipeline, runtime
from scholar_lens.src.guide_pipeline import guide_branch_id, guide_title
from scholar_lens.src.jobs import JobRequest
from scholar_lens.src.metrics import TokenUsageTracker
from scholar_lens.src.post_conventions import BlogPost, load_local_conventions
from scholar_lens.src.publisher import Publisher
from scholar_lens.src.runtime import RunContext
from scholar_lens.src.tech_guide import TechGuide
from tests.test_publisher import PNG, FakeGithub, FakeRepo


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
    out = guide_document(document, guide, korean=True)
    assert "[Image:" not in out
    assert f"]({arch})" in out
    assert "logo.svg" not in out


def test_branch_id_counts_reference_material() -> None:
    docs = ["https://docs.vllm.ai/en/latest/"]
    assert guide_branch_id(docs) != guide_branch_id(
        [*docs, "https://arxiv.org/abs/2309.06180"]
    )
    # A guide from reference material alone still gets its own branch.
    assert guide_branch_id(["https://blog.x.io/a"]) != guide_branch_id([])


LONG = (
    "vLLM: Installation, Offline and Online Serving, PagedAttention and "
    "Continuous Batching Internals, Configuration and Deployment"
)


@pytest.mark.parametrize(
    ("title", "seo_title", "expected"),
    [
        ("vLLM Serving Guide", "", "vLLM Serving Guide"),
        (LONG, "vLLM: Serving and Internals", "vLLM: Serving and Internals"),
        # An over-long seo_title: the title cut at a word boundary, never
        # reduced to the tool's name before the colon.
        (
            LONG,
            "x" * 71,
            "vLLM: Installation, Offline and Online Serving, PagedAttention and",
        ),
        (
            LONG,
            "",
            "vLLM: Installation, Offline and Online Serving, PagedAttention and",
        ),
    ],
)
def test_a_guide_title_over_the_seo_limit_is_replaced(
    title: str, seo_title: str, expected: str
) -> None:
    assert guide_title(title, seo_title, 70) == expected


class FakeGuideGenerator:
    """The guide generator with its model calls replaced: each run gives the
    guide a new title and proposes a new topic."""

    def __init__(self, title: str, topic: str) -> None:
        self.title, self.topic = title, topic
        self.notes: list[str] = []

    async def generate(self, urls: list[str], **_: Any) -> TechGuide:
        body = "## 설치\n\n![구조](https://arxiv.org/html/2309.06180v1/x1.png)\n\n설명입니다.\n"
        return TechGuide(
            topic=self.title,
            body=body,
            source_urls=list(urls),
            image_urls=["https://arxiv.org/html/2309.06180v1/x1.png"],
        )

    async def classify(self, guide: TechGuide, **_: Any) -> dict[str, str]:
        return {
            "category": self.topic,
            "keywords": "vLLM",
            "controlled_tags": "Efficient-Attention",
        }


async def _run_guide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    context: RunContext,
    generator: FakeGuideGenerator,
) -> Any:
    async def no_guideline(_: RunContext) -> None:
        return None

    monkeypatch.setattr(guide_pipeline, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(guide_pipeline, "TechGuideGenerator", lambda **_: generator)
    monkeypatch.setattr(
        guide_pipeline,
        "WebResearcher",
        lambda **_: SimpleNamespace(close=lambda: None, search_notes=lambda: []),
    )
    monkeypatch.setattr(guide_pipeline, "load_translation_guideline", no_guideline)
    request = JobRequest(kind="guide", urls=[DOCS], create_pr=False)
    return await guide_pipeline.run_guide_pipeline(
        context, request, TokenUsageTracker()
    )


DOCS = "https://docs.vllm.ai/en/latest/"
_GITHUB = Github(
    enabled=True,
    repo_name="owner/blog",
    author_name="Bot",
    author_email="bot@example.com",
)


async def test_a_retitled_rerun_updates_the_published_guide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guide's identity is its sources (the branch), not the model's
    title: a rerun under a new title and topic replaces the published guide,
    keeping its file name, URL (date, topic) and image names."""
    spec = load_local_conventions()
    repo = FakeRepo({})
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    monkeypatch.setattr(FakeGithub, "repo", repo)
    monkeypatch.setattr("scholar_lens.src.publisher.Github", FakeGithub)
    monkeypatch.setattr(
        runtime,
        "build_publisher",
        lambda ctx: Publisher(
            _GITHUB,
            root_dir=tmp_path,
            conventions=spec,
            site_posts=ctx.blog_posts,
            fetch_image=lambda url: PNG,
        ),
    )
    session = boto3.Session(region_name="us-east-1")

    def context(posts: list[BlogPost]) -> RunContext:
        return RunContext(
            config=Config(),
            default_boto_session=session,
            bedrock_boto_session=session,
            conventions=spec,
            blog_posts=posts,
            blog_tags={},
        )

    first = await _run_guide(
        tmp_path / "1",
        monkeypatch,
        context([]),
        FakeGuideGenerator("vLLM Serving Guide", "Language-Models"),
    )
    # Merged a while ago: the post is on the blog, published by the branch.
    published = f"_posts/2026-01-05-{first.markdown_path.name[11:]}"
    repo.merge(f"paper-reviews/{guide_branch_id([DOCS])}", [published])
    blog = BlogPost(
        path=published,
        date=datetime(2026, 1, 5, 3, tzinfo=UTC),
        categories=["Tech Guides", "Language-Models"],
    )
    second = await _run_guide(
        tmp_path / "2",
        monkeypatch,
        context([blog]),
        FakeGuideGenerator("Serving LLMs with vLLM", "Finetuning"),
    )
    post = second.markdown_path.read_text()
    assert second.markdown_path.name == "2026-01-05-vllm-serving-guide.md"
    assert 'title: "Serving LLMs with vLLM"' in post
    assert "\ndate: 2026-01-05 03:00:00\n" in post
    assert 'categories: ["Tech Guides", "Language-Models"]' in post
    assert "cover: /assets/images/language-models.jpg" in post
    assert "](/assets/images/vllm-serving-guide-1.png)" in post
