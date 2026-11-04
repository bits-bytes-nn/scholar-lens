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
    GuideSettings,
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
    assert strip_markers("기본값은 10입니다 [[E2, E3]].") == "기본값은 10입니다."
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
        raise AssertionError(schema)

    async def text(self, *, role, system, prompt) -> str:
        self.calls.append(f"text:{role}")
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
