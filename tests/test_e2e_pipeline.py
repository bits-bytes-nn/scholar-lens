"""End-to-end pipeline runs with the LLM stages faked (no AWS, no network).

The fakes return what real model output tends to look like — legacy math
delimiters, glued display math, escaped underscores in math, an arXiv image
URL, a leaked wrapper tag — so these tests prove the deterministic layer turns
such output into a post that passes the blog conventions, and that a post it
cannot repair is blocked before anything is published.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import boto3
import pytest

from scholar_lens.configs import Config
from scholar_lens.src import guide_pipeline, paper_pipeline, publisher, runtime
from scholar_lens.src.content_extractor import Attributes
from scholar_lens.src.explainer import Paper
from scholar_lens.src.jobs import JobRequest
from scholar_lens.src.metrics import TokenUsageTracker
from scholar_lens.src.parser import Content, Figure
from scholar_lens.src.post_conventions import load_local_conventions
from scholar_lens.src.post_validator import PostValidationError, validate_post
from scholar_lens.src.tech_guide import TechGuide

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
ARXIV_FIGURE = "https://arxiv.org/html/2106.09685v2/x1.png"

SUMMARY_FROM_MODEL = r"""> **요약:** LoRA는 사전 학습 가중치를 고정하고 저랭크 행렬만 학습합니다.

## 🔍 이 연구가 왜 필요한가요?
전체 파인튜닝은 과제마다 \(W_0\) 크기의 사본을 저장해야 합니다.

## 💡 어떤 해결책을 제시하나요?
가중치 변화량을 두 행렬의 곱으로 제한합니다.
$$h = W\_0 x + B A x$$
여기서 \(r \ll \min(d, k)\)입니다.

![LoRA 재매개변수화](https://arxiv.org/html/2106.09685v2/x1.png)

아래 그림은 동결 경로와 학습 경로를 보여줍니다.

## ⚙️ 어떻게 구현했나요?
\[
\Delta W = B A
\]
$A$는 가우시안, $B$는 0으로 초기화합니다.

## 📊 핵심 실험 결과는 무엇인가요?

| 모델 | 학습 파라미터 | MNLI |
| --- | --- | --- |
| RoBERTa-base (LoRA) | 0.3M | 87.5 |

## 🔮 의의와 한계는 무엇인가요?
배치 안에서 서로 다른 과제를 섞기 어렵다는 한계가 있습니다.</summary>
"""


def _paper(**attribute_overrides: object) -> Paper:
    attributes = {
        "affiliation": "Microsoft",
        "category": "Finetuning",
        "keywords": ["Low-Rank Adaptation", "Parameter Efficient Fine Tuning"],
        "controlled_tags": ["Quantization"],
    }
    attributes.update(attribute_overrides)
    return Paper(
        arxiv_id="2106_09685",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        authors=["Edward J. Hu", "Yelong Shen"],
        published=datetime(2021, 6, 17, 17, 37, 18),
        pdf_url="https://arxiv.org/pdf/2106.09685",
        content=Content(text="Low-rank adaptation (LoRA) freezes W_0 ... quantization"),
        attributes=Attributes(**attributes),
        figures=[Figure(figure_id="1", path=ARXIV_FIGURE, caption="Figure 1")],
    )


@pytest.fixture
def pipeline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """A RunContext with local conventions, no S3/GitHub, faked image fetch."""
    monkeypatch.setattr(paper_pipeline, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(guide_pipeline, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(runtime, "ROOT_DIR", tmp_path)
    fetched: list[str] = []

    def build_publisher(context):  # type: ignore[no-untyped-def]
        def fetch(url: str) -> bytes:
            fetched.append(url)
            return PNG

        return publisher.Publisher(
            context.config.resources.github,
            root_dir=tmp_path,
            conventions=context.require_conventions(),
            fetch_image=fetch,
        )

    monkeypatch.setattr(runtime, "build_publisher", build_publisher)
    context = runtime.RunContext(
        config=Config(),
        default_boto_session=boto3.Session(region_name="us-west-2"),
        bedrock_boto_session=boto3.Session(region_name="us-west-2"),
        conventions=load_local_conventions(),
    )
    return context, fetched


def _fake_paper_data(paper: Paper):  # type: ignore[no-untyped-def]
    async def prepare(*args, **kwargs):  # type: ignore[no-untyped-def]
        return paper, None, None

    return prepare


def _fake_summarizer(output: str):  # type: ignore[no-untyped-def]
    class FakeSummarizer:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.notes: list[str] = []

        async def summarize(self, paper: Paper) -> dict[str, str]:
            return {
                "summary": output,
                "subtitle": "LoRA: 대규모 언어 모델의 저랭크 적응",
                "urls": "[LoRA code](https://github.com/microsoft/LoRA)",
            }

    return FakeSummarizer


class TestSummaryEndToEnd:
    async def test_messy_model_output_becomes_a_valid_post(
        self, pipeline_env, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
    ) -> None:
        context, fetched = pipeline_env
        monkeypatch.setattr(
            paper_pipeline, "_prepare_paper_data", _fake_paper_data(_paper())
        )
        monkeypatch.setattr(
            paper_pipeline, "PaperSummarizer", _fake_summarizer(SUMMARY_FROM_MODEL)
        )

        outcome = await paper_pipeline.run_paper_pipeline(
            context,
            JobRequest(kind="summarize", source="2106.09685"),
            TokenUsageTracker(),
        )

        post = Path(outcome.markdown_path).read_text(encoding="utf-8")
        violations = validate_post(post, context.conventions, "summarize")
        assert [v for v in violations if v.level == "error"] == []
        front = post.split("\n---\n", 1)[0]
        assert 'paper_author: "Microsoft"' in front and "\nauthor:" not in front
        assert 'categories: ["Paper Summaries", "Finetuning"]' in front
        assert "cover: /assets/images/finetuning.jpg" in front
        assert '"Quantization"' in front and '"Low-Rank-Adaptation"' in front
        assert "seo_title:" not in front  # title is short enough
        assert "use_math: true" in front
        body = post.split("\n---\n", 1)[1]
        assert r"\(" not in body and r"\[" not in body and r"W\_0" not in body
        assert "\n\n$$h = W_0 x + B A x$$\n\n" in body
        assert "</summary>" not in body
        assert fetched == [ARXIV_FIGURE]
        assert "](/assets/images/" in body and "arxiv.org/html" not in body

    async def test_post_without_controlled_tag_is_blocked(
        self, pipeline_env, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
    ) -> None:
        context, _ = pipeline_env
        paper = _paper(
            controlled_tags=[],
            keywords=["Low-Rank Adaptation"],
            category="Training & Inference",
        )
        monkeypatch.setattr(
            paper_pipeline, "_prepare_paper_data", _fake_paper_data(paper)
        )
        monkeypatch.setattr(
            paper_pipeline, "PaperSummarizer", _fake_summarizer(SUMMARY_FROM_MODEL)
        )

        with pytest.raises(PostValidationError) as info:
            await paper_pipeline.run_paper_pipeline(
                context,
                JobRequest(kind="summarize", source="2106.09685"),
                TokenUsageTracker(),
            )
        rules = {v.rule for v in info.value.violations}
        assert {"tags", "categories"} <= rules
        work_dir = Path(paper_pipeline.ROOT_DIR) / "papers" / "2106_09685"
        assert not list(work_dir.glob("20*.md"))  # no post written
        assert list(work_dir.glob("*.rejected.md"))


class TestGuideEndToEnd:
    async def test_guide_gets_spec_taxonomy_and_validates(
        self, pipeline_env, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
    ) -> None:
        context, _ = pipeline_env
        guide = TechGuide(
            topic="Serving LLMs with vLLM",
            body=(
                "# vLLM 가이드\n\n## PagedAttention 이해하기\n\nKV 캐시를 페이지 단위로 관리합니다.\n"
                "블록 크기 \\(B\\)를 정합니다.\n\n![arch](https://docs.vllm.ai/arch.png)\n"
            ),
            source_urls=["https://docs.vllm.ai/en/latest/"],
            image_urls=["https://docs.vllm.ai/arch.png"],  # offered to the writer
        )

        class FakeGenerator:
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                pass

            async def generate(self, *args, **kwargs) -> TechGuide:  # type: ignore[no-untyped-def]
                return guide

            async def classify(self, *args, **kwargs) -> dict[str, str]:  # type: ignore[no-untyped-def]
                return {
                    "category": "Language Models",
                    "controlled_tags": "Efficient-Attention, Not-On-The-List",
                    "keywords": "Paged Attention, Continuous Batching",
                    "seo_title": "",
                }

        monkeypatch.setattr(guide_pipeline, "TechGuideGenerator", FakeGenerator)
        monkeypatch.setattr(guide_pipeline, "WebResearcher", MagicMock())

        outcome = await guide_pipeline.run_guide_pipeline(
            context,
            JobRequest(kind="guide", urls=["https://docs.vllm.ai/en/latest/"]),
            TokenUsageTracker(),
        )
        post = Path(outcome.markdown_path).read_text(encoding="utf-8")
        violations = validate_post(post, context.conventions, "guide")
        assert [v for v in violations if v.level == "error"] == []
        # External doc images stay links but are flagged for review.
        assert any(v.rule == "image" for v in violations if v.level == "warning")
        front = post.split("\n---\n", 1)[0]
        assert 'categories: ["Tech Guides", "Language-Models"]' in front
        assert "paper_author" not in front and "\nauthor:" not in front
        assert '"Paged-Attention"' in front and '"Efficient-Attention"' in front
        assert "Not-On-The-List" not in front
        assert "use_math: true" in front and "$$B$$" in post
        assert "\n# " not in post.split("\n---\n", 1)[1]


REVIEW_BODY_FROM_MODEL = r"""# LoRA: 대규모 언어 모델의 저랭크 적응

## 문제 정의
전체 파인튜닝은 \(\Phi_0 + \Delta\Phi\)를 과제마다 저장합니다.

## 제안 방법
$$h = W\_0 x + \Delta W x = W_0 x + B A x$$
초기에는 \(B = 0\)이므로 \(\Delta W = 0\)입니다.

![LoRA 구조](https://arxiv.org/html/2106.09685v2/x1.png)

아래 그림에서 동결 경로와 학습 경로가 합쳐집니다.

| 모델 | 학습 파라미터 | MNLI |
| --- | --- | --- |
| RoBERTa-base (LoRA) | 0.3M | 87.5 |
</explanation>"""


class TestReviewEndToEnd:
    async def test_review_layout_and_conventions(
        self, pipeline_env, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
    ) -> None:
        from scholar_lens.src.explainer import ReviewText

        context, _ = pipeline_env

        class FakeReviewer:
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                pass

            async def run(self) -> ReviewText:
                return ReviewText(
                    body=REVIEW_BODY_FROM_MODEL,
                    key_takeaways=(
                        "### 왜 필요한가요?\n- 파인튜닝 사본 저장 비용이 큽니다.\n\n"
                        "### 무엇을 제안하나요?\n- 저랭크 분해 \\(BA\\)만 학습합니다."
                    ),
                    closing="## 한계와 시사점\n\n배치 안에서 과제를 섞기 어렵습니다.",
                    subtitle="",
                )

        async def prepare(*args, **kwargs):  # type: ignore[no-untyped-def]
            return _paper(), None, MagicMock()

        monkeypatch.setattr(paper_pipeline, "_prepare_paper_data", prepare)
        monkeypatch.setattr(paper_pipeline, "PaperReviewer", FakeReviewer)

        outcome = await paper_pipeline.run_paper_pipeline(
            context, JobRequest(kind="review", source="2106.09685"), TokenUsageTracker()
        )
        post = Path(outcome.markdown_path).read_text(encoding="utf-8")
        assert [
            v
            for v in validate_post(post, context.conventions, "review")
            if v.level == "error"
        ] == []
        front, body = post.split("\n---\n", 1)
        # The body's H1 (translated title) becomes the subtitle, not a heading.
        assert 'subtitle: "LoRA: 대규모 언어 모델의 저랭크 적응"' in front
        assert 'categories: ["Paper Reviews", "Finetuning"]' in front
        assert body.lstrip().startswith("## TL;DR\n\n### 왜 필요한가요?")
        assert (
            body.index("## 문제 정의")
            < body.index("## 한계와 시사점")
            < body.index("### References")
        )
        assert "\n# " not in body and "</explanation>" not in body
        assert r"\(" not in body and "W\\_0" not in body


async def test_tags_the_paper_never_mentions_become_pr_notes(
    pipeline_env, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
) -> None:
    context, _ = pipeline_env
    paper = _paper(keywords=["Low-Rank Adaptation", "Direct Preference Optimization"])
    monkeypatch.setattr(paper_pipeline, "_prepare_paper_data", _fake_paper_data(paper))
    monkeypatch.setattr(
        paper_pipeline, "PaperSummarizer", _fake_summarizer(SUMMARY_FROM_MODEL)
    )
    captured: list = []
    real = runtime.publish_artifact

    async def spy(context, request, *, create_pr):  # type: ignore[no-untyped-def]
        captured.append(request)
        return await real(context, request, create_pr=create_pr)

    monkeypatch.setattr(paper_pipeline, "publish_artifact", spy)
    await paper_pipeline.run_paper_pipeline(
        context, JobRequest(kind="summarize", source="2106.09685"), TokenUsageTracker()
    )
    notes = captured[0].review_notes
    assert any("Direct-Preference-Optimization" in n for n in notes)
    assert not any("Low-Rank-Adaptation" in n for n in notes)


async def test_degraded_output_reaches_pr_notes_and_job_warnings(
    pipeline_env, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
) -> None:
    context, _ = pipeline_env
    monkeypatch.setattr(
        paper_pipeline, "_prepare_paper_data", _fake_paper_data(_paper())
    )
    fake = _fake_summarizer(SUMMARY_FROM_MODEL)

    class NotingSummarizer(fake):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            self.notes = ["Summary is still long (4,000 Hangul characters)."]

    monkeypatch.setattr(paper_pipeline, "PaperSummarizer", NotingSummarizer)
    captured: list = []
    real = runtime.publish_artifact

    async def spy(context, request, *, create_pr):  # type: ignore[no-untyped-def]
        captured.append(request)
        return await real(context, request, create_pr=create_pr)

    monkeypatch.setattr(paper_pipeline, "publish_artifact", spy)
    outcome = await paper_pipeline.run_paper_pipeline(
        context, JobRequest(kind="summarize", source="2106.09685"), TokenUsageTracker()
    )
    assert any("still long" in w for w in outcome.warnings)
    assert any("still long" in n for n in captured[0].review_notes)


class _FakeS3:
    def __init__(self, result: object) -> None:
        self.result = result
        self.keys: list[str] = []

    def get_text(self, key: str) -> str | None:
        self.keys.append(key)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result  # type: ignore[return-value]


def _guideline_context(s3: _FakeS3):  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    return SimpleNamespace(
        s3_handler=s3,
        config=SimpleNamespace(resources=SimpleNamespace(s3_prefix="sl")),
    )


async def test_translation_guideline_is_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scholar_lens.src.aws_helpers import S3OperationError

    monkeypatch.setattr(paper_pipeline, "ROOT_DIR", tmp_path)
    load = paper_pipeline._load_translation_guideline
    missing = _FakeS3(None)
    assert await load(_guideline_context(missing)) is None  # type: ignore[arg-type]
    assert missing.keys and missing.keys[0].startswith("sl/")
    found = _FakeS3('[{"term": "rank", "rendering": "랭크"}]')
    assert await load(_guideline_context(found)) == [  # type: ignore[arg-type]
        {"term": "rank", "rendering": "랭크"}
    ]
    with pytest.raises(S3OperationError):
        await load(_guideline_context(_FakeS3(S3OperationError("denied"))))  # type: ignore[arg-type]


async def test_guideline_failure_stops_before_paid_stages(
    pipeline_env, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
) -> None:
    context, _ = pipeline_env
    prepare = MagicMock()

    async def broken(_context):  # type: ignore[no-untyped-def]
        raise RuntimeError("s3 down")

    monkeypatch.setattr(paper_pipeline, "_load_translation_guideline", broken)
    monkeypatch.setattr(paper_pipeline, "_prepare_paper_data", prepare)
    with pytest.raises(RuntimeError, match="s3 down"):
        await paper_pipeline.run_paper_pipeline(
            context,
            JobRequest(kind="summarize", source="2106.09685"),
            TokenUsageTracker(),
        )
    prepare.assert_not_called()


async def test_guide_budget_abort_keeps_the_generator_notes(
    pipeline_env, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
) -> None:
    from scholar_lens.src.metrics import TokenBudgetExceeded

    context, _ = pipeline_env

    class FakeGenerator:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.notes = ["Token budget reached: later sections were not refined."]

        async def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise TokenBudgetExceeded("4600000 > 4500000")

    monkeypatch.setattr(guide_pipeline, "TechGuideGenerator", FakeGenerator)
    monkeypatch.setattr(guide_pipeline, "WebResearcher", MagicMock())
    with pytest.raises(TokenBudgetExceeded) as caught:
        await guide_pipeline.run_guide_pipeline(
            context,
            JobRequest(kind="guide", urls=["https://docs.vllm.ai/en/latest/"]),
            TokenUsageTracker(),
        )
    assert caught.value.run_notes == FakeGenerator().notes  # type: ignore[attr-defined]
