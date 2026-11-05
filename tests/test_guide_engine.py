"""The evidence-based guide engine: page evidence with code and settings, the
authority rule, the guide outline's order, the guide-only draft checks and a
scripted run from reading to the assembled body."""

from __future__ import annotations

import asyncio
from typing import Any

from scholar_lens.application.evidence_writer import (
    ClaimCheck,
    EvidenceCheck,
    SectionReview,
    SentenceVerdict,
)
from scholar_lens.application.guide_engine import (
    EvidenceGuideEngine,
    GuideReview,
    GuideSettings,
    ReviewFinding,
    authority_rule,
)
from scholar_lens.application.ports import AgentLimits, AgentOutcome
from scholar_lens.domain.evidence import (
    EvidenceLedger,
    SourceDocument,
    evidence_problems,
)
from scholar_lens.domain.guide.evidence_guide import (
    GuideOutline,
    guide_draft_problems,
    guide_outline_problems,
    repair_guide_outline,
    strip_markers,
)
from scholar_lens.src.guide_writer import _source_urls, guide_documents
from scholar_lens.src.web_research import OfficialDocs, PageContent

DOC_TEXT = """## Overview

Serve batches requests on each replica to raise throughput.

## Configuration

Set `max_batch_size` to cap a batch. The default value of max_batch_size is 10.

```python
@serve.batch(max_batch_size=8)
async def handle(self, requests):
    return [r * 2 for r in requests]
```

## Architecture

The controller schedules replicas and the proxy routes each request to a replica.
"""

BLOG_TEXT = """## Why batching

Batching amortizes the model's fixed cost over many requests, so the GPU stays busy.
A batch that waits too long adds latency to every request in it.
"""


def _official() -> SourceDocument:
    return SourceDocument.from_markdown(
        "D1",
        "Ray Serve batching",
        "full_text",
        DOC_TEXT,
        figures=[("https://docs.ray.io/arch.png", "Serve architecture")],
        url="https://docs.ray.io/serve/batching.html",
        authority="official",
    )


def _blog() -> SourceDocument:
    return SourceDocument.from_markdown(
        "W1",
        "Batching explained",
        "full_text",
        BLOG_TEXT,
        url="https://example.com/batching",
        authority="third_party",
    )


# ---- evidence ---------------------------------------------------------------


def test_code_must_be_the_pages_code() -> None:
    doc = _official()
    good = evidence_problems(
        doc,
        kind="code",
        claim="배치 크기를 8로 제한하는 핸들러",
        quote="",
        anchor="S2",
        code="@serve.batch(max_batch_size=8)\nasync def handle(self, requests):",
    )
    assert good == []
    invented = evidence_problems(
        doc,
        kind="code",
        claim="배치 핸들러",
        quote="",
        anchor="S2",
        code="@serve.batch(max_batch_size=8, timeout=1)",
    )
    assert any("not verbatim" in p for p in invented)


def test_a_setting_needs_its_scope_and_names_only_quoted_identifiers() -> None:
    doc = _official()
    problems = evidence_problems(
        doc,
        kind="setting",
        claim="`max_batch_size` 기본값은 10, `batch_wait_timeout_s`와 함께 씁니다",
        quote="The default value of max_batch_size is 10.",
        anchor="S2",
    )
    assert any("scope" in p for p in problems)
    assert any("batch_wait_timeout_s" in p for p in problems)


def test_settings_come_from_official_pages_only() -> None:
    rule = authority_rule(has_official=True)
    assert rule(_blog(), "setting", {})
    assert rule(_blog(), "method", {}) == []
    assert rule(_official(), "setting", {}) == []
    older = _blog().model_copy(update={"authority": "older_docs"})
    assert authority_rule(has_official=False)(older, "setting", {})


# ---- outline ----------------------------------------------------------------


def _ledger() -> EvidenceLedger:
    ledger = EvidenceLedger()
    doc, blog = _official(), _blog()
    ledger.add(
        doc,
        kind="finding",
        claim="레플리카마다 요청을 묶어 처리량을 높입니다",
        quote="Serve batches requests on each replica to raise throughput.",
        anchor="S1",
    )
    ledger.add(
        doc,
        kind="setting",
        claim="`max_batch_size` 기본값은 10입니다",
        quote="The default value of max_batch_size is 10.",
        anchor="S2",
        scope="@serve.batch 데코레이터",
    )
    ledger.add(
        doc,
        kind="method",
        claim="컨트롤러가 레플리카를 배치하고 프록시가 요청을 보냅니다",
        quote="The controller schedules replicas and the proxy routes each request to a replica.",
        anchor="S3",
    )
    ledger.add(
        doc, kind="figure", claim="Serve 구조", quote="", anchor="S3", figure_id="F1"
    )
    ledger.add(
        blog,
        kind="limitation",
        claim="오래 기다리는 배치는 지연을 늘립니다",
        quote="A batch that waits too long adds latency to every request in it.",
        anchor="S1",
    )
    return ledger


def _outline(**changes: Any) -> GuideOutline:
    data: dict[str, Any] = {
        "title": "Ray Serve Batching from the Inside",
        "running_example": "번역 API",
        "code_language": "python",
        "sections": [
            {
                "kind": "tldr",
                "title": "요약",
                "thesis": "t",
                "depth": "standard",
                "evidence": ["E1"],
            },
            {
                "kind": "overview",
                "title": "처리량을 묶어 올리는 배치",
                "thesis": "t",
                "evidence": ["E1"],
            },
            {
                "kind": "architecture",
                "title": "컨트롤러와 프록시가 나누는 일",
                "thesis": "t",
                "depth": "deep",
                "evidence": ["E3", "E4"],
            },
            {
                "kind": "usage",
                "title": "데코레이터 하나로 켜는 배치",
                "thesis": "t",
                "evidence": ["E2"],
            },
            {
                "kind": "choosing",
                "title": "지연과 맞바꾸는 처리량",
                "thesis": "t",
                "evidence": ["E5"],
            },
        ],
    }
    data.update(changes)
    return GuideOutline.model_validate(data)


def _documents() -> dict[str, SourceDocument]:
    return {"D1": _official(), "W1": _blog()}


def test_a_good_outline_has_no_problems_after_repair() -> None:
    outline, notes = repair_guide_outline(_outline(), _ledger())
    tldr = outline.sections[0]
    assert (tldr.title, tldr.depth, tldr.evidence) == ("TL;DR", "brief", [])
    assert outline.sections[1].evidence == ["E1"] and notes == []
    problems = guide_outline_problems(
        outline, _ledger(), _documents(), title_max_length=70, max_sections=14
    )
    assert problems == []
    twice = [s.model_dump() for s in _outline().sections]
    twice[3]["evidence"] = ["E2", "E3"]
    repaired, notes = repair_guide_outline(_outline(sections=twice), _ledger())
    assert repaired.sections[3].evidence == ["E2"] and any("twice" in n for n in notes)


def test_the_outline_order_and_required_kinds_are_checked() -> None:
    sections = _outline().sections
    shuffled = _outline(
        sections=[
            s.model_dump() for s in [sections[0], sections[1], sections[3], sections[4]]
        ]
    )
    problems = guide_outline_problems(
        shuffled, _ledger(), _documents(), title_max_length=10, max_sections=14
    )
    assert any("architecture" in p for p in problems)
    assert any("title is longer" in p for p in problems)
    late = _outline(
        sections=[
            s.model_dump()
            for s in [sections[0], sections[1], sections[3], sections[2], sections[4]]
        ]
    )
    repaired, notes = repair_guide_outline(late, _ledger())
    assert [s.kind for s in repaired.sections] == [
        "tldr",
        "overview",
        "architecture",
        "usage",
        "choosing",
    ]
    assert "sections sorted into the guide's order" in notes


def test_unused_reference_material_is_a_problem() -> None:
    reference = _blog().model_copy(update={"authority": "reference"})
    sections = [s.model_dump() for s in _outline().sections]
    sections[-1]["evidence"] = ["E2"]
    sections[3]["evidence"] = []
    problems = guide_outline_problems(
        _outline(sections=sections),
        _ledger(),
        {"D1": _official(), "W1": reference},
        title_max_length=70,
        max_sections=14,
    )
    assert any("reference material W1" in p for p in problems)


# ---- drafts -----------------------------------------------------------------


def test_guide_draft_problems() -> None:
    long_code = "```python\n" + "x = 1\n" * 45 + "```"
    draft = (
        "## 데코레이터 하나로 켜는 배치\n\n"
        "공식 문서에 따르면 기본값은 10입니다 [[E2]].\n\n"
        f"{long_code}\n\n```typescript\nconst a = 1;\n```\n"
    )
    problems = guide_draft_problems(draft, code_language="python")
    assert any("names the page" in p for p in problems)
    assert any("45-line code block" in p for p in problems)
    assert any("show it in python" in p for p in problems)
    assert guide_draft_problems("## 배치\n\n기본값은 10입니다 [[E2]].\n") == []


def test_markers_are_stripped_and_cited_sources_listed() -> None:
    assert strip_markers("기본값은 10입니다 [[E2, E3]].") == "기본값은 10입니다.\n"
    urls = _source_urls(
        [_blog(), _official()],
        ["https://docs.ray.io/serve/batching.html", "https://x.dev/ref"],
    )
    assert urls == [
        "https://docs.ray.io/serve/batching.html",
        "https://example.com/batching",
        "https://x.dev/ref",
    ]


def test_pages_become_documents_by_authority() -> None:
    official = OfficialDocs.from_urls(
        ["https://docs.ray.io/en/latest/serve/index.html"]
    )
    pages = [
        PageContent(
            url="https://docs.ray.io/en/latest/serve/batching.html",
            title="Batching",
            text=DOC_TEXT,
        ),
        PageContent(url="https://example.com/b", title="Blog", text=BLOG_TEXT * 3),
        PageContent(
            url="https://arxiv.org/abs/2309.06180",
            title="Paper",
            text=BLOG_TEXT * 3,
            user_reference=True,
        ),
        PageContent(url="https://example.com/stub", title="Stub", text="menu"),
    ]
    docs = guide_documents(pages, official)
    assert [(d.paper_id, d.authority) for d in docs] == [
        ("D1", "official"),
        ("W1", "third_party"),
        ("R1", "reference"),
    ]


# ---- a scripted run -----------------------------------------------------------


class ScriptedModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_agent(
        self, *, role, system, prompt, tools, limits: AgentLimits
    ) -> AgentOutcome:
        self.calls.append("read")
        record = {t.__name__: t for t in tools}["record_evidence"]
        if "Page D1" in prompt:
            record(
                "finding",
                "레플리카마다 요청을 묶어 처리량을 높입니다",
                "Serve batches requests on each replica to raise throughput.",
                "S1",
            )
            assert record(
                "setting",
                "`max_batch_size` 기본값은 10입니다",
                "The default value of max_batch_size is 10.",
                "S2",
                "@serve.batch",
            ).startswith("recorded")
            record(
                "method",
                "컨트롤러가 레플리카를 배치하고 프록시가 요청을 보냅니다",
                "The controller schedules replicas and the proxy routes each request to a replica.",
                "S3",
            )
            record("figure", "Serve 구조", "", "S3", "", "", "F1")
            record(
                "code",
                "배치 크기를 8로 제한하는 핸들러",
                "",
                "S2",
                code="@serve.batch(max_batch_size=8)\nasync def handle(self, requests):\n    return [r * 2 for r in requests]",
            )
        else:
            # A setting from a third-party page is rejected.
            assert record(
                "setting",
                "기본값은 10입니다",
                "Batching amortizes the model's fixed cost over many requests, so the GPU stays busy.",
                "S1",
                "Serve",
            ).startswith("REJECTED")
            record(
                "limitation",
                "오래 기다리는 배치는 지연을 늘립니다",
                "A batch that waits too long adds latency to every request in it.",
                "S1",
            )
        return AgentOutcome(text="done", stop_reason="end_turn")

    async def structured(self, *, role, system, prompt, schema):
        self.calls.append(schema.__name__)
        if schema is GuideOutline:
            sections = [s.model_dump() for s in _outline().sections]
            sections[3]["evidence"] = ["E2", "E5"]
            sections[4]["evidence"] = ["E6"]
            return _outline(sections=sections)
        if schema is ClaimCheck:
            bad = "기본값은 20" in prompt
            return ClaimCheck(
                verdicts=[SentenceVerdict(n=1, verdict="wrong", reason="10")]
                if bad
                else []
            )
        if schema is SectionReview:
            return SectionReview()
        if schema is EvidenceCheck:
            return EvidenceCheck()
        if schema is GuideReview:
            assert "<guide>" in prompt and "@serve.batch" in prompt
            return GuideReview(
                trace="request -> handler -> result",
                issues=[
                    ReviewFinding(
                        section="데코레이터 하나로 켜는 배치",
                        problem="serve is not imported.",
                        fix="from ray import serve",
                    )
                ],
            )
        raise AssertionError(schema)

    async def text(self, *, role, system, prompt) -> str:
        self.calls.append(f"text:{role}")
        if "serve is not imported" in prompt:
            self.calls.append("code-fix")
            assert "<source_sections>" in prompt
            return "## 데코레이터 하나로 켜는 배치\n\n`max_batch_size` 기본값은 10입니다 [[E2]].\n\n```python\nfrom ray import serve\n\n@serve.batch(max_batch_size=8)\n```\n"
        if "title: 데코레이터" in prompt:
            # The writer sees the docs section its evidence comes from.
            assert "[D1 S2: Ray Serve batching | Configuration | official]" in prompt
        if "<problems>" in prompt:
            title = prompt.split("## ", 1)[1].split("\n", 1)[0]
            return f"## {title}\n\n`max_batch_size` 기본값은 10입니다 [[E2]].\n\n```python\n@serve.batch(max_batch_size=8)\n```\n"
        if "kind: tldr" in prompt:
            return "## TL;DR\n\nServe는 레플리카마다 요청을 묶어 처리량을 높입니다 [[E1]].\n"
        title = prompt.split("title: ", 1)[1].split("\n", 1)[0]
        if title.startswith("데코레이터"):
            return f"## {title}\n\n`max_batch_size` 기본값은 20입니다 [[E2]].\n\n```python\n@serve.batch(max_batch_size=8)\n```\n"
        if title.startswith("컨트롤러"):
            return f"## {title}\n\n컨트롤러가 레플리카를 배치하고 프록시가 요청을 보냅니다 [[E3]].\n\n![구조](https://docs.ray.io/arch.png)\n\n| 구성요소 | 역할 |\n| --- | --- |\n| 컨트롤러 | 배치 |\n"
        if title.startswith("처리량"):
            return f"## {title}\n\n요청을 묶으면 처리량이 오릅니다 [[E1]].\n"
        return f"## {title}\n\n오래 기다리는 배치는 지연을 늘립니다 [[E6]].\n"


def test_a_scripted_guide_run() -> None:
    model = ScriptedModel()
    engine = EvidenceGuideEngine(
        model,
        GuideSettings(
            language="Korean",
            title_max_length=70,
            reader_tag="sonnet",
            min_total_chars=5_000,
        ),
        identifier_check=lambda md: ["--made-up"] if "--made-up" in md else [],
    )
    guide = asyncio.run(engine.write("Ray Serve batching", [_blog(), _official()]))
    body = guide.body
    # The docs are read first; their ids number the ledger.
    assert guide.ledger.get("E2").kind == "setting"
    assert body.startswith("## TL;DR")
    assert (
        body.index("## TL;DR")
        < body.index("## 처리량")
        < body.index("## 컨트롤러")
        < body.index("## 데코레이터")
        < body.index("## 지연")
    )
    assert "기본값은 10입니다" in body and "20" not in body
    assert "[[E" not in body and "![구조](https://docs.ray.io/arch.png)" in body
    assert [d.paper_id for d in guide.cited] == ["D1", "W1"]
    # The TL;DR is written last.
    assert [c for c in model.calls if c.startswith("text")][-1] == "text:write"
    assert model.calls.count("read") == 2
    assert "code-fix" in model.calls and "from ray import serve" in body


def test_a_figure_without_a_caption_is_placed_with_the_text_before_it() -> None:
    text = (
        "## Architecture\n\nServe runs a controller and one proxy per node.\n\n"
        '[Image: alt="../_images/architecture-2.0.svg", src="https://x/a.svg"] '
        "(https://x/a.svg)\n\n## Autoscaling\n\nIt adds replicas.\n"
    )
    doc = SourceDocument.from_markdown(
        "D1",
        "Architecture",
        "full_text",
        text,
        figures=[("https://x/a.svg", "../_images/architecture-2.0.svg")],
    )
    (figure,) = doc.figures
    assert (figure.caption, figure.anchor) == ("", "S1")
    assert (
        figure.context
        == "Architecture: Serve runs a controller and one proxy per node."
    )
    assert "F1 | figure in S1 | text before it: Architecture:" in doc.outline()


def test_table_rows_are_claims_and_a_wrong_row_leaves_whole() -> None:
    from scholar_lens.domain.evidence_draft import claim_units, drop_sentences

    md = "## 결정\n\n두 결정이 있습니다.\n\n| 결정 | 의미 |\n| --- | --- |\n| Deny | 막음 |\n| Transform | 바꿈 |\n"
    assert claim_units(md)[1:] == ["| Deny | 막음 |", "| Transform | 바꿈 |"]
    assert drop_sentences(md, ["| Transform | 바꿈 |"]).endswith("| Deny | 막음 |\n")


def test_writer_notes_and_empty_headings_are_problems_and_removed() -> None:
    from scholar_lens.domain.evidence_draft import (
        draft_problems,
        drop_empty_headings,
        strip_revision_notes,
    )

    md = (
        "## 스티어링\n\n## 스티어링 핸들러\n\n핸들러는 도구 호출을 안내합니다 [[E1]].\n\n"
        "참고: 요청하신 import 줄은 넣지 않았습니다.\n"
    )
    problems = draft_problems(
        md,
        title="스티어링",
        own=["E1"],
        earlier=(),
        ledger=_ledger(),
        earlier_numbers={},
        figure_urls=(),
        free_numbers="",
    )
    assert any("note about the writing" in p for p in problems)
    assert any("nothing under it" in p for p in problems)
    assert drop_empty_headings(md).startswith("## 스티어링 핸들러")
    assert "요청하신" not in strip_revision_notes(md)
    english = "## A\n\n내용입니다.\n\nNotes on fixes I could not make fully: none.\n"
    assert "Notes" not in strip_revision_notes(english)


def test_the_guides_own_view_is_marked_only_where_it_chooses() -> None:
    from scholar_lens.domain.evidence_draft import draft_problems

    md = "## 배치\n\n이 글의 판단으로는 배치를 먼저 켭니다.\n"
    kwargs: dict[str, Any] = {
        "title": "배치",
        "own": (),
        "earlier": (),
        "ledger": _ledger(),
        "earlier_numbers": {},
        "figure_urls": (),
        "free_numbers": "",
    }
    assert any("own view" in p for p in draft_problems(md, own_view_limit=0, **kwargs))
    assert not any("own view" in p for p in draft_problems(md, **kwargs))


def test_a_fact_two_pages_state_is_kept_from_the_official_page() -> None:
    from scholar_lens.domain.evidence import duplicate_evidence

    ledger = EvidenceLedger()
    blog, doc = _blog(), _official()
    ledger.add(
        blog,
        kind="finding",
        claim="배치는 고정 비용을 여러 요청에 나눕니다",
        quote="Batching amortizes the model's fixed cost over many requests, so the GPU stays busy.",
        anchor="S1",
    )
    ledger.add(
        doc,
        kind="finding",
        claim="배치는 고정 비용을 여러 요청에 나눕니다.",
        quote="Serve batches requests on each replica to raise throughput.",
        anchor="S1",
    )
    assert duplicate_evidence(ledger, {"D1": 0, "W1": 2}) == {"E1": "E2"}


def test_code_defined_earlier_lists_the_running_examples_definitions() -> None:
    from scholar_lens.application.guide_engine import defined_in_code

    md = (
        "## A\n\n```python\n@serve.deployment\nclass Translator:\n"
        "    def translate(self, text: str) -> str:\n        return text\n```\n"
    )
    assert defined_in_code([md]).split("\n") == [
        "@serve.deployment",
        "class Translator:",
        "def translate(self, text: str) -> str:",
    ]


def test_a_marker_standing_for_a_word_and_hedges_are_problems() -> None:
    draft = (
        "## 설치\n\n표의 근거는 [[E3]]입니다.\n\n"
        "import 경로는 쓰는 SDK 버전에서 확인해야 합니다 [[E2]].\n"
    )
    problems = guide_draft_problems(draft)
    assert any("stands for a word" in p for p in problems)
    assert any("hedges" in p for p in problems)


def test_stripping_markers_leaves_no_empty_heading_or_marker_line() -> None:
    body = "## A\n\n[[E1]]\n\n## B\n\n내용입니다 [[E2]].\n"
    assert strip_markers(body) == "## B\n\n내용입니다.\n"


def test_sources_are_listed_once_however_spelled() -> None:
    page = _official().model_copy(
        update={"url": "https://docs.ray.io/serve/batching.html"}
    )
    assert _source_urls([page], ["https://docs.ray.io/serve/batching.html/"]) == [
        "https://docs.ray.io/serve/batching.html"
    ]


def test_a_reader_that_skips_a_required_figure_is_asked_once_more() -> None:
    from scholar_lens.application.evidence_writer import EvidenceWriter
    from scholar_lens.application.guide_engine import GUIDE_PROMPTS

    prompts: list[str] = []

    class SkipsFigures:
        async def run_agent(self, *, role, system, prompt, tools, limits):
            prompts.append(prompt)
            record = {t.__name__: t for t in tools}["record_evidence"]
            if len(prompts) == 2:
                record("figure", "Serve 구조", "", "S3", "", "", "F1")
            return AgentOutcome(text="", stop_reason="end_turn")

    writer = EvidenceWriter(SkipsFigures(), GUIDE_PROMPTS, max_fix_rounds=0)
    items = asyncio.run(
        writer.read_document(
            _official(),
            prompt="read",
            limits=AgentLimits(turns=5, total_tokens=1000),
            required=("figure",),
        )
    )
    assert [e.kind for e in items] == ["figure"]
    assert len(prompts) == 2 and "['figure'] is still missing" in prompts[1]


def test_empty_headings_respect_levels_and_code() -> None:
    from scholar_lens.domain.markdown.lint import drop_empty_headings

    md = (
        "## A\n\n### A1\n\n본문입니다.\n\n## B\n\n\n## C\n\n```md\n## not a heading\n```\n"
        "\n## D\n"
    )
    assert drop_empty_headings(md) == (
        "## A\n\n### A1\n\n본문입니다.\n\n## C\n\n```md\n## not a heading\n```\n"
    )


def test_a_figure_is_placed_by_its_id_whatever_anchor_the_reader_gives() -> None:
    doc = _official()
    assert (
        evidence_problems(
            doc, kind="figure", claim="구조도", quote="", anchor="F1", figure_id="F1"
        )
        == []
    )
    ledger = EvidenceLedger()
    item = ledger.add(
        doc, kind="figure", claim="구조도", quote="", anchor="F1", figure_id="F1"
    )
    assert item.figure_url == "https://docs.ray.io/arch.png"


def test_third_party_pages_do_not_name_identifiers_when_docs_exist() -> None:
    rule = authority_rule(has_official=True)
    claim = {"claim": "`vllm:gpu_cache_usage_perc`로 캐시 사용률을 봅니다"}
    assert rule(_blog(), "finding", claim)
    assert rule(_official(), "finding", claim) == []
    assert authority_rule(has_official=False)(_blog(), "finding", claim) == []
    assert rule(_blog(), "code", {"claim": "배포"})
    assert rule(_blog().model_copy(update={"authority": "reference"}), "code", {}) == []


def test_a_figure_without_a_quote_takes_the_text_before_it() -> None:
    text = (
        "## Results\n\nThroughput with three parallel outputs on A100.\n\n"
        '[Image: alt="", src="https://x/n3.png"]\n'
    )
    doc = SourceDocument.from_markdown(
        "R1", "Blog", "full_text", text, figures=[("https://x/n3.png", "")]
    )
    item = EvidenceLedger().add(
        doc, kind="figure", claim="처리량", quote="", anchor="S1", figure_id="F1"
    )
    assert "three parallel outputs" in item.quote


def test_internal_source_ids_in_the_text_are_problems() -> None:
    assert any(
        "internal id" in p
        for p in guide_draft_problems("## A\n\n영상 튜토리얼(W8)이 보여 줍니다.\n")
    )
    assert guide_draft_problems("## A\n\n`D1` 등급의 GPU를 씁니다.\n") == []


def test_evidence_for_another_sdk_language_leaves_a_python_guide() -> None:
    from scholar_lens.domain.guide.evidence_guide import (
        drop_other_language,
        other_language_only,
    )

    assert other_language_only("TypeScript 전용, 미들웨어 등록", "python")
    assert other_language_only("only in TypeScript", "py")
    assert not other_language_only("Python 전용", "python")
    assert not other_language_only("TypeScript only", "")
    ledger = _ledger()
    ledger.items[1] = ledger.items[1].model_copy(
        update={"scope": "TypeScript 전용 @serve.batch"}
    )
    outline, dropped = drop_other_language(_outline(), ledger)
    assert dropped == ["E2"] and outline.sections[3].evidence == []


def test_product_imports_the_official_pages_never_show_are_flagged() -> None:
    from scholar_lens.domain.guide.evidence_guide import unverified_imports

    official = (
        "from strands import Agent\nfrom strands.vended_tools import file_editor\n"
    )
    code = (
        "```python\nfrom strands import Agent\nfrom strands_tools import file_editor\n"
        "from pydantic import BaseModel\nimport strands.vended_tools\n```\n"
    )
    assert unverified_imports(code, official, ["Strands", "strands-agents"]) == [
        "strands_tools"
    ]
    assert unverified_imports(code, "", ["strands"]) == []


def test_a_sentence_crediting_a_source_must_cite_it() -> None:
    from scholar_lens.domain.guide.evidence_guide import misattributed, source_aliases

    blog = _blog().model_copy(update={"url": "https://agents-lib.com/compare/ray"})
    aliases = source_aliases([_official(), blog], products=["Ray Serve", "ray"])
    assert aliases == {"W1": ["agents-lib"]}
    ledger = _ledger()
    bad = "## 운영\n\nagents-lib에 따르면 GPU 레플리카는 PENDING에 머뭅니다 [[E3]].\n"
    good = (
        "## 운영\n\nagents-lib는 오래 기다리는 배치가 지연을 늘린다고 봅니다 [[E5]].\n"
    )
    assert misattributed(bad, ledger, aliases)
    assert misattributed(good, ledger, aliases) == []


def test_internal_source_ids_become_titles_outside_code() -> None:
    from scholar_lens.domain.guide.evidence_guide import name_sources

    md = "W2 페이지는 드리프트를 봅니다.\n\n```python\nW2 = 1\n```\n\n`D1` 등급\n"
    assert name_sources(md, {"W2": "Ray Serve vs Triton"}) == (
        "Ray Serve vs Triton 페이지는 드리프트를 봅니다.\n\n```python\nW2 = 1\n```\n\n`D1` 등급\n"
    )
