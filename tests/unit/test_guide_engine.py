"""The evidence-based guide engine: page evidence with code and settings, the
authority rule, the guide outline's order, the guide-only draft checks and a
scripted run from reading to the assembled body."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, get_args

import pytest

from scholar_lens.application.evidence_writer import (
    ClaimCheck,
    Edit,
    EvidenceCheck,
    FixedSection,
    PostFinding,
    PostReview,
    SectionChecks,
    SectionEdits,
    SectionReview,
)
from scholar_lens.application.guide_engine import (
    EvidenceGuideEngine,
    GuideSettings,
)
from scholar_lens.application.ports import AgentLimits, AgentOutcome
from scholar_lens.application.use_cases.guide import _source_urls, guide_documents
from scholar_lens.domain.coverage import (
    CoverageTarget,
    ExtractedTarget,
    ExtractedTargets,
)
from scholar_lens.domain.evidence import (
    AUTHORITY_RANK,
    Authority,
    EvidenceLedger,
    SourceAuthority,
    SourceDocument,
    authority_rank,
    evidence_problems,
)
from scholar_lens.domain.guide.evidence_guide import (
    GuideOutline,
    guide_draft_problems,
    guide_outline_problems,
    repair_guide_outline,
    strip_markers,
)
from scholar_lens.domain.post_spec import evidence_policy
from scholar_lens.domain.research import PageContent
from scholar_lens.domain.sources import SourceSet
from scholar_lens.domain.specs import GUIDE_SPEC
from scholar_lens.domain.verifiers import no_checks
from tests.fakes import claim_check, evidence_check, with_context

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


def authority_rule(has_official: bool):  # type: ignore[no-untyped-def]
    """The guide's evidence policy for a corpus with or without official
    docs."""
    corpus = {"official"} if has_official else set()
    return lambda document, kind, fields: evidence_policy(
        GUIDE_SPEC.evidence, document, kind, fields, corpus=corpus
    )


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
        "기본값은 10입니다 [[E2]].\n\n"
        f"{long_code}\n\n```typescript\nconst a = 1;\n```\n"
    )
    problems = guide_draft_problems(draft, code_language="python")
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
    sources = SourceSet.from_urls(["https://docs.ray.io/en/latest/serve/index.html"])
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
    sources.record_all(pages)
    docs = guide_documents(pages)
    assert [(d.paper_id, d.authority) for d in docs] == [
        ("src-D1", "official"),
        ("src-W1", "third_party"),
        ("src-R1", "reference"),
    ]


# ---- a scripted run -----------------------------------------------------------


class ScriptedModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_agent(
        self, *, role, system, prompt, tools, limits: AgentLimits
    ) -> AgentOutcome:
        if "<already_recorded>" in prompt:  # the follow-up of a short read
            self.calls.append("follow-up")
            return AgentOutcome(text="", stop_reason="end_turn")
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

    async def structured(self, *, role, system, prompt, schema, context=""):
        if schema is SectionEdits:  # a fix as one edit: the old section for the new
            old = prompt.split("<section>\n", 1)[1].split("\n</section>", 1)[0]
            new = await self.text(
                role=role, system=system, prompt=prompt, context=context
            )
            return SectionEdits(edits=[Edit(find=old.strip(), replace=new.strip())])
        if schema is FixedSection:  # a fix answers like a text call, structured
            section = await self.text(
                role=role, system=system, prompt=prompt, context=context
            )
            return FixedSection(section=section, notes="고친 내용: 수치를 바로잡음")
        prompt = with_context(context, prompt)
        self.calls.append(schema.__name__)
        if schema is ExtractedTargets:
            # The key-target extraction reads the official page (E3).
            assert "[D1] Ray Serve batching (official)" in prompt
            return ExtractedTargets(
                targets=[
                    ExtractedTarget(
                        statement="max_batch_size caps a batch", source="D1"
                    )
                ]
            )
        if schema is GuideOutline:
            sections = [s.model_dump() for s in _outline().sections]
            sections[3]["evidence"] = ["E2", "E5"]
            sections[4]["evidence"] = ["E6"]
            return _outline(sections=sections, coverage={"T1": ["E2"]})
        if schema is ClaimCheck:
            bad = "기본값은 20" in prompt
            return claim_check(
                prompt, lambda n, _s: ("wrong", "10") if bad and n == 1 else None
            )
        if schema is SectionReview:
            return SectionReview()
        if schema is EvidenceCheck:
            return evidence_check(prompt)
        if schema is PostReview and "repetition only" in system:
            # A read of its own, its findings fixed with the code review's.
            self.calls.append("repetition-review")
            return PostReview(
                issues=[
                    PostFinding(
                        section="처리량을 묶어 올리는 배치",
                        problem="It repeats what the TL;DR teaches.",
                        fix="Refer back to it.",
                    )
                ]
            )
        if schema is PostReview:
            # The consistency read (E3: one generic read with the spec's code
            # trace, the post under <post>).
            assert "<post>" in prompt and "@serve.batch" in prompt
            assert "<source_code>" in prompt
            return PostReview(
                trace="request -> handler -> result",
                issues=[
                    PostFinding(
                        section="데코레이터 하나로 켜는 배치",
                        problem="serve is not imported.",
                        fix="from ray import serve",
                    )
                ],
            )
        raise AssertionError(schema)

    async def text(self, *, role, system, prompt, context="") -> str:
        prompt = with_context(context, prompt)
        self.calls.append(f"text:{role}")
        if "serve is not imported" in prompt:
            self.calls.append("code-fix")
            assert "<source_sections>" in prompt
            return "## 데코레이터 하나로 켜는 배치\n\n`max_batch_size` 기본값은 10입니다 [[E2]].\n\n```python\nfrom ray import serve\n\n@serve.batch(max_batch_size=8)\n```\n"
        if "title: 데코레이터" in prompt:
            # The writer sees the docs section its evidence comes from.
            assert "[D1 S2: Ray Serve batching | Configuration | official]" in prompt
        if "It repeats what the TL;DR teaches" in prompt:
            self.calls.append("repetition-fix")
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
    # The scripted drafts are a sentence each: no minimum length (its fixes
    # are scripted for other problems).
    engine.limits = dataclasses.replace(engine.limits, min_length_share=0.0)
    guide = asyncio.run(engine.write("Ray Serve batching", [_blog(), _official()]))
    # Each page's reader recorded fewer than the low end of its count
    # (the spec's main or minor range) and was continued once.
    assert model.calls.count("follow-up") == model.calls.count("read") == 2
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
    assert "repetition-review" in model.calls and "repetition-fix" in model.calls


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


def test_empty_headings_are_problems_and_note_blocks_are_removed() -> None:
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
    assert any("nothing under it" in p for p in problems)
    assert drop_empty_headings(md).startswith("## 스티어링 핸들러")
    english = "## A\n\n내용입니다.\n\n## Changes made\n\n- fixed E2\n"
    assert "Changes" not in strip_revision_notes(english)


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
        for p in guide_draft_problems("## A\n\n영상 튜토리얼(src-W8)이 보여 줍니다.\n")
    )
    # Product and model names never take the id form.
    assert (
        guide_draft_problems("## A\n\nDeepSeek R1 모델과 W8 양자화를 씁니다.\n") == []
    )


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


def test_internal_source_ids_become_titles_outside_code() -> None:
    from scholar_lens.domain.guide.evidence_guide import name_sources

    md = (
        "src-W2 페이지는 드리프트를 봅니다. W2 양자화도 씁니다.\n\n"
        "```python\nsrc-W2 = 1\n```\n\n`src-D1` 등급\n"
    )
    assert name_sources(md, {"src-W2": "Serving Compared"}) == (
        "Serving Compared 페이지는 드리프트를 봅니다. W2 양자화도 씁니다.\n\n"
        "```python\nsrc-W2 = 1\n```\n\n`src-D1` 등급\n"
    )


def test_dropping_a_sentence_leaves_code_and_tables_alone() -> None:
    from scholar_lens.domain.evidence_draft import drop_sentences

    md = (
        "## A\n\n틀린 문장입니다. 맞는 문장입니다.\n\n```yaml\napplications:\n"
        "  - name: app1\n    import_path: text_ml:app\n```\n\n"
        "| 키 | 값 |\n| --- | --- |\n| a | 1 |\n\n- 첫째\n"
    )
    out = drop_sentences(md, ["틀린 문장입니다."])
    assert "  - name: app1\n    import_path: text_ml:app\n" in out
    assert "| a | 1 |" in out and "맞는 문장입니다." in out


def test_the_projects_own_repository_counts_as_official() -> None:
    requested = ["https://docs.ray.io/en/latest/serve/index.html"]
    edit = "https://github.com/ray-project/ray/edit/master/doc/serve.md"
    pages = [
        PageContent(
            url=f"https://docs.ray.io/en/latest/serve/p{n}.html",
            title="Docs",
            text=DOC_TEXT,
            links=[edit, *(["https://github.com/someone/fork"] if n == 0 else [])],
        )
        for n in range(3)
    ] + [
        PageContent(
            url="https://github.com/ray-project/ray/blob/master/doc/source/serve/architecture.md",
            title="architecture.md",
            text=BLOG_TEXT * 3,
        ),
        PageContent(
            url="https://github.com/someone/fork", title="Fork", text=BLOG_TEXT * 3
        ),
    ]
    # Decided at gather time: the repositories the docs pages link to.
    SourceSet.from_urls(requested).with_repositories(pages).record_all(pages)
    docs = guide_documents(pages)
    assert [d.authority for d in docs[3:]] == ["official", "third_party"]
    assert [d.source_authority for d in docs[3:]] == ["official_repo", "third_party"]
    assert docs[3].version == "master"


def test_install_commands_are_checked_against_the_package_extras() -> None:
    from scholar_lens.domain.guide.evidence_guide import (
        install_requirements,
        unknown_extras,
    )

    md = '```bash\npip install "strands-agents[cedar,otel]" pydantic -U\nuv pip install ray[serve]\n```\n'
    assert install_requirements(md) == [
        ("strands-agents", frozenset({"cedar", "otel"})),
        ("pydantic", frozenset()),
        ("ray", frozenset({"serve"})),
    ]
    provided = {"strands-agents": frozenset({"cedar"}), "ray": frozenset({"serve"})}
    assert unknown_extras(md, provided) == ["strands-agents[otel]"]


def test_guide_python_is_checked_as_pasted_in_order() -> None:
    from scholar_lens.domain.guide.evidence_guide import python_code_problems

    sections = [
        (
            "기본",
            "## 기본\n\n```python\nfrom strands import Agent\n\ndef build(t):\n    return Agent(t)\n\ncedar = 1\n```\n",
            True,
        ),
        (
            "서비스",
            "## 서비스\n\n```python\ndef handle(x):\n    return build(x), cedar, ask(x)\n```\n\n```python\nkey = <your key>\n```\n\n```bash\necho $UNDEFINED\n```\n",
            True,
        ),
    ]
    problems = python_code_problems(sections)
    assert [t for t, _ in problems] == ["서비스", "서비스"]
    assert "ask" in problems[0][1] and "build" not in problems[0][1]
    assert "does not parse" in problems[1][1]


def test_internals_excerpts_sessions_and_star_imports_are_not_paste_problems() -> None:
    from scholar_lens.domain.guide.evidence_guide import python_code_problems

    excerpt = "## 구조\n\n```python\nasync def cycle(agent):\n    return _handle(agent)\n```\n"
    repl = "## 사용\n\n```python\n>>> agent(prompt)\n```\n\n```python\n%pip install x\n```\n"
    star = "## 사용\n\n```python\nfrom x import *\nrun(Agent())\n```\n"
    assert python_code_problems([("구조", excerpt, False)]) == []
    assert python_code_problems([("사용", repl, True)]) == []
    assert python_code_problems([("사용", star, True)]) == []


def test_definitions_in_list_items_sessions_and_bad_blocks_still_count() -> None:
    from scholar_lens.domain.guide.evidence_guide import (
        guide_draft_problems,
        python_code_problems,
    )

    setup = (
        "## 준비\n\n1. 에이전트를 만듭니다.\n\n   ```python\n   agent = Agent()\n   ```\n\n"
        "```python\n>>> helper = make()\n```\n\n"
        "```python\ndef invoke_async(prompt, *, ...) -> Result: ...\nretry = 3\n```\n"
    )
    use = "## 사용\n\n```python\nfrom strands import Agent\nprint(agent('hi'), helper, retry)\n```\n"
    assert python_code_problems([("준비", setup, False), ("사용", use, True)]) == []
    later = "## 나중\n\n```python\nprint(missing_name)\n```\n"
    problems = python_code_problems(
        [("준비", setup, False), ("사용", use, True), ("나중", later, True)]
    )
    assert problems and "missing_name" in problems[0][1]
    code_marker = "## A\n\n설명입니다 [[E1]].\n\n```python\nx = 1  # [[E2]]\n```\n"
    assert any("inside code" in p for p in guide_draft_problems(code_marker))


def test_evidence_for_another_sdk_is_recognised_by_its_field() -> None:
    from scholar_lens.domain.guide.evidence_guide import other_sdk

    assert other_sdk("typescript", "python")
    assert other_sdk("ts", "py")
    assert not other_sdk("python", "python")
    assert not other_sdk("", "python") and not other_sdk("typescript", "")


def test_a_topic_is_owned_by_its_first_section() -> None:
    from scholar_lens.domain.guide.evidence_guide import repair_guide_outline

    sections = [s.model_dump() for s in _outline().sections]
    sections[1]["topics"] = ["Structured output"]
    sections[2]["topics"] = ["structured  output", "Hooks"]
    outline, notes = repair_guide_outline(_outline(sections=sections), _ledger())
    taught = [s.topics for s in outline.sections]
    assert ["Structured output"] in taught and ["Hooks"] in taught
    assert any("kept in its first section" in n for n in notes)


# ---- the evidence cache key ---------------------------------------------------


class _DictStore:
    def __init__(self) -> None:
        self.blobs: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.blobs.get(key)

    def put(self, key: str, text: str) -> None:
        self.blobs[key] = text


class _Reader:
    """A reader that records one finding of the official page."""

    def __init__(self) -> None:
        self.reads = 0

    async def run_agent(self, *, prompt, tools, **_: Any) -> AgentOutcome:  # type: ignore[no-untyped-def]
        if "<already_recorded>" in prompt:  # the follow-up of a short read
            return AgentOutcome(text="", stop_reason="end_turn")
        self.reads += 1
        record = {t.__name__: t for t in tools}["record_evidence"]
        record(
            "finding",
            "레플리카마다 요청을 묶습니다",
            "Serve batches requests on each replica to raise throughput.",
            "S1",
        )
        return AgentOutcome(text="done", stop_reason="end_turn")


def test_cached_evidence_is_reused_only_for_the_same_reader_inputs() -> None:
    model = _Reader()
    store = _DictStore()
    engine = EvidenceGuideEngine(
        model,  # type: ignore[arg-type]
        GuideSettings(language="Korean", title_max_length=70, reader_tag="t"),
        store,
    )
    page = SourceDocument.from_markdown(
        "D1", "Ray Serve batching", "full_text", DOC_TEXT, authority="official"
    )
    asyncio.run(engine.read("Serve batching", [page]))
    asyncio.run(engine.read("Serve batching", [page]))
    assert model.reads == 1
    # Another topic, or other key targets, is another reader prompt.
    asyncio.run(engine.read("Serve autoscaling", [page]))
    assert model.reads == 2
    engine.targets = [CoverageTarget(id="T1", statement="a batch cap", source="D1")]
    asyncio.run(engine.read("Serve autoscaling", [page]))
    assert model.reads == 3
    asyncio.run(engine.read("Serve autoscaling", [page]))
    assert model.reads == 3 and len(store.blobs) == 3


# ---- the one authority ranking -------------------------------------------------


@pytest.mark.parametrize(
    "vocabulary", [Authority, SourceAuthority], ids=["engine", "source"]
)
def test_every_authority_has_one_rank(vocabulary: object) -> None:
    for value in get_args(vocabulary):
        assert authority_rank(value) == AUTHORITY_RANK.get(
            value, AUTHORITY_RANK["third_party"]
        )
    assert authority_rank("") == authority_rank("not-an-authority")
    assert authority_rank("") == AUTHORITY_RANK["third_party"]


@pytest.mark.parametrize("unknown", ["", "reference", "older_docs"])
def test_cited_sources_and_passages_sort_by_the_one_ranking(unknown: str) -> None:
    docs = [
        SourceDocument.from_markdown(
            f"D{k}",
            "t",
            "full_text",
            "## A\n\nText.\n",
            url=f"https://x.dev/{k}",
            authority=a,  # type: ignore[arg-type]
        )
        for k, a in enumerate(["older_docs", unknown, "third_party", "official"])
    ]
    urls = _source_urls(docs, [])
    assert urls == [
        d.url for d in sorted(docs, key=lambda d: authority_rank(d.authority))
    ]
    engine = EvidenceGuideEngine(
        ScriptedModel(),  # type: ignore[arg-type]
        GuideSettings(language="Korean", title_max_length=70, reader_tag="t"),
    )
    engine._documents = {d.paper_id: d for d in docs}
    assert engine.source_rank() == {
        d.paper_id: authority_rank(d.authority) for d in docs
    }


# ---- the whole-post fix and the program a reader pastes ----------------------

_DEFINE = "client = Client()"
_EXAMPLE = f"```python\nfrom pkg import Client\n\n{_DEFINE}\n```"


class _RepetitionFix:
    """The whole-post reads: the repetition read names the first usage
    section; its fix drops the running example's definition (the block
    still parses, the section's own checks pass)."""

    def __init__(self) -> None:
        self.fixes = 0

    async def structured(self, *, system, prompt, schema, **_: Any):  # type: ignore[no-untyped-def]
        if schema is PostReview and "repetition only" in system:
            return PostReview(
                issues=[
                    PostFinding(
                        section="클라이언트 만들기",
                        problem="It repeats the overview.",
                        fix="Say it once.",
                    )
                ]
            )
        if schema is PostReview:
            return PostReview()
        if schema is SectionEdits:
            self.fixes += 1
            return SectionEdits(edits=[Edit(find=f"\n{_DEFINE}", replace="")])
        if schema is ClaimCheck:
            return claim_check(prompt, lambda _n, _s: None)
        if schema is EvidenceCheck:
            return evidence_check(prompt)
        raise AssertionError(schema)


@pytest.mark.parametrize("runnable", [True, False], ids=["runnable", "no-runnable"])
def test_a_whole_post_fix_that_undefines_a_later_usage_name_is_refused(
    runnable: bool,
) -> None:
    """The guide's runnable sections build one program: a repetition fix
    that drops a name a later usage section runs is not taken. A spec
    without runnable kinds has no program to keep."""
    model = _RepetitionFix()
    engine = EvidenceGuideEngine(
        model,  # type: ignore[arg-type]
        GuideSettings(language="Korean", title_max_length=70, reader_tag="t"),
    )
    if not runnable:
        engine.spec = dataclasses.replace(engine.spec, runnable_kinds=frozenset())
    ledger = _ledger()
    outline = _outline()
    usage = outline.sections[3]
    define = usage.model_copy(update={"title": "클라이언트 만들기"})
    use = usage.model_copy(update={"title": "클라이언트 실행"})
    written = [
        (
            define,
            "## 클라이언트 만들기\n\n`max_batch_size` 기본값은 10입니다 [[E2]].\n\n"
            + _EXAMPLE
            + "\n",
        ),
        (
            use,
            "## 클라이언트 실행\n\n`max_batch_size` 기본값은 10입니다 [[E2]].\n\n"
            "```python\nclient.run()\n```\n",
        ),
    ]
    assert engine.program_check(written) == []
    for section, _md in written:
        engine.checked[id(section)] = SectionChecks(
            no_checks(), tuple(ledger), (), "", False, ""
        )
    out = asyncio.run(engine.review_post(outline, ledger, written))
    assert model.fixes == 1
    assert (out[0][1] == written[0][1]) is runnable
    refused = [
        n
        for n in engine.notes
        if "left as they were" in n and "adds problems" in n and "client" in n
    ]
    assert bool(refused) is runnable


# ---- the TL;DR's length ---------------------------------------------------------

_COPIED = "컨트롤러가 레플리카를 배치하고 프록시가 요청을 보냅니다 [[E3]]."


class _ShortTldr:
    """A TL;DR draft well below its target that copies a body sentence;
    its fix deletes the copy."""

    def __init__(self, draft: str) -> None:
        self.draft = draft
        self.fixes: list[str] = []

    async def text(self, **_: Any) -> str:
        return self.draft

    async def structured(self, *, prompt, schema, **_: Any):  # type: ignore[no-untyped-def]
        if schema is SectionEdits:
            self.fixes.append(prompt)
            return SectionEdits(edits=[Edit(find=f" {_COPIED}", replace="")])
        if schema is ClaimCheck:
            return claim_check(prompt, lambda _n, _s: None)
        if schema is SectionReview:
            return SectionReview()
        if schema is EvidenceCheck:
            return evidence_check(prompt)
        raise AssertionError(schema)


def test_a_short_tldr_has_no_minimum_and_may_be_shortened() -> None:
    """The TL;DR is as long as what it sums up: a draft below the share of
    a target at or above the floor is not too short, and a fix that cuts a
    sentence of it is taken."""
    opening = " ".join(
        [
            "Serve는 레플리카마다 요청을 묶어 처리량을 높입니다 [[E1]].",
            "요청을 하나씩 처리하면 모델의 고정 비용이 요청마다 듭니다 [[E1]].",
            "묶어 보내면 그 비용을 여러 요청이 나눠 냅니다 [[E1]].",
            "그래서 같은 레플리카로 더 많은 요청을 받습니다 [[E1]].",
        ]
    )
    draft = f"## TL;DR\n\n{opening} {_COPIED}\n"
    model = _ShortTldr(draft)
    engine = EvidenceGuideEngine(
        model,  # type: ignore[arg-type]
        GuideSettings(language="Korean", title_max_length=70, reader_tag="t"),
    )
    target = engine.limits.min_length_floor
    engine.limits = dataclasses.replace(engine.limits, short_draft_share=0.0)
    outline = _outline()
    written = [
        (outline.sections[2], f"## 컨트롤러와 프록시가 나누는 일\n\n{_COPIED}\n"),
    ]
    tldr = asyncio.run(
        engine.write_tldr(outline, _ledger(), outline.sections[0], target, written)
    )
    # The draft is below the minimum share of its target.
    assert len(opening) < engine.limits.min_length_share * target
    assert model.fixes and all("characters of prose" not in p for p in model.fixes)
    assert "copies a sentence of the body" in model.fixes[0]
    assert tldr == f"## TL;DR\n\n{opening}\n"
    assert not any("characters of prose" in n for n in engine.notes)
