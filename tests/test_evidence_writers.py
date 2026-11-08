"""The glue from a run to the evidence engines (src/guide_writer.py,
src/paper_writer.py): which model and fallback each role gets, the cost
ceiling, the evidence cache, and the shape of what the publisher receives.
The engines and Bedrock are replaced; no network."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import pytest

from scholar_lens.adapters.bedrock import strands_models
from scholar_lens.adapters.bedrock.strands_models import RoleModel
from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
from scholar_lens.adapters.web.research import OfficialDocs, PageContent, ResearchCorpus
from scholar_lens.application import guide_engine, paper_engine
from scholar_lens.application.paper_engine import EvidencePaper
from scholar_lens.configs import Config
from scholar_lens.domain.evidence import EvidenceLedger, SourceDocument
from scholar_lens.domain.paper.models import Attributes, Content, Paper, ReviewText
from scholar_lens.src import evidence_models, guide_writer, paper_writer, runtime
from scholar_lens.src.tech_guide import GuideMaterial

FIGURE = "https://docs.x.dev/img/arch.png"
BROKEN = "https://docs.x.dev/img/gone.png"


@dataclass
class Seen:
    roles: dict[str, RoleModel] = field(default_factory=dict)
    ceilings: list[float | None] = field(default_factory=list)
    settings: list[Any] = field(default_factory=list)
    stores: list[Any] = field(default_factory=list)
    root: Path = Path()


@pytest.fixture
def seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Seen:
    seen = Seen()

    class FakeModel:
        def __init__(self, session, roles, record) -> None:  # type: ignore[no-untyped-def]
            seen.roles = roles

    def role_model(context, model_id, effort, thinking):  # type: ignore[no-untyped-def]
        return RoleModel(
            invoke_id=model_id, model_label=model_id, effort=effort, thinking=thinking
        )

    def budgeted(tracker, ceiling):  # type: ignore[no-untyped-def]
        seen.ceilings.append(ceiling)
        return tracker.record

    async def unservable(urls):  # type: ignore[no-untyped-def]
        return [u for u in urls if u == BROKEN]

    monkeypatch.setattr(strands_models, "StrandsLanguageModel", FakeModel)
    monkeypatch.setattr(evidence_models, "role_model", role_model)
    monkeypatch.setattr(evidence_models, "budgeted", budgeted)
    monkeypatch.setattr(evidence_models, "unservable", unservable)
    seen.root = tmp_path
    return seen


def _context(root: Path) -> runtime.RunContext:
    return runtime.RunContext(
        config=Config(),
        root_dir=root,
        default_boto_session=boto3.Session(region_name="us-west-2"),
        bedrock_boto_session=boto3.Session(region_name="us-west-2"),
    )


# ---- guide ---------------------------------------------------------------------


class FakeGenerator:
    official = OfficialDocs.from_urls(["https://docs.x.dev/guide/"])
    products = ["xserve"]
    notes = ["gather note"]

    async def gather(self, urls, **_kwargs):  # type: ignore[no-untyped-def]
        page = PageContent(
            url="https://docs.x.dev/guide/start",
            title="Start &amp; run",
            text="xserve runs models. " * 40,
            final_url="https://docs.x.dev/guide/start/",
            authority="official",
        )
        return GuideMaterial(topic="xserve", corpus=ResearchCorpus(pages=[page]))


def test_a_guide_run_wires_roles_ceiling_cache_and_sources(
    seen: Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeGuideEngine:
        def __init__(
            self,
            model,
            settings,
            *,
            store,
            checkpoints,
            identifier_check,
            products,  # type: ignore[no-untyped-def]
            package_facts,
            sdk=None,
        ) -> None:
            seen.settings.append(settings)
            seen.stores.append(store)
            # The checkpoints share the reading cache's directory and S3
            # prefix (a Batch retry reads both back).
            assert checkpoints is store
            self.check = identifier_check

        async def write(self, topic, documents, *, guideline):  # type: ignore[no-untyped-def]
            (doc,) = documents
            assert doc.paper_id == "src-D1" and doc.authority == "official"
            ledger = EvidenceLedger(
                items=[_figure(doc, FIGURE, "E1"), _figure(doc, BROKEN, "E2")]
            )
            body = f"## xserve\n\n![a]({FIGURE})\n\n![b]({BROKEN})\n"
            return guide_engine.EvidenceGuide(
                title="xserve 가이드",
                body=body,
                outline=None,  # type: ignore[arg-type]
                ledger=ledger,
                cited=[doc],
                notes=["engine note"],
            )

    monkeypatch.setattr(guide_engine, "EvidenceGuideEngine", FakeGuideEngine)
    monkeypatch.setattr(guide_writer, "docs_packages", lambda _text: {})
    context = _context(seen.root)
    guide = asyncio.run(
        guide_writer.write_guide_with_evidence(
            context,
            FakeGenerator(),  # type: ignore[arg-type]
            urls=["https://docs.x.dev/guide/start"],
            reference_urls=[],
            discover_subpages=False,
            search_queries=None,
            translation_guideline=None,
            tracker=TokenUsageTracker(),
            title_max_length=60,
        )
    )
    cfg = context.config.tech_guide
    assert seen.roles["read"].invoke_id == cfg.synopsis_model_id
    assert seen.roles["write"].invoke_id == cfg.writing_model_id
    assert seen.roles["write"].fallback is not None
    assert seen.roles["write"].fallback.invoke_id == cfg.synopsis_model_id
    assert seen.roles["evaluate"].invoke_id == cfg.evaluation_model_id
    assert seen.ceilings == [cfg.max_cost_usd]
    assert seen.stores[0].local_dir.name == "guide-evidence"
    assert seen.stores[0].local_dir.is_relative_to(seen.root)
    assert seen.settings[0].max_sources == cfg.max_sources_read
    # Served address, unescaped title, notes, the broken figure left out.
    assert guide.source_urls == ["https://docs.x.dev/guide/start/"]
    assert guide.source_titles == {"https://docs.x.dev/guide/start/": "Start & run"}
    assert guide.notes[:2] == ["gather note", "engine note"]
    assert any(BROKEN in n for n in guide.notes)
    assert guide.image_urls == [FIGURE]


def _figure(doc: SourceDocument, url: str, eid: str = "E1"):  # type: ignore[no-untyped-def]
    from scholar_lens.domain.evidence import Evidence

    return Evidence(
        id=eid,
        paper_id=doc.paper_id,
        kind="figure",
        claim="구조",
        quote="q",
        anchor="S1",
        figure_url=url,
    )


# ---- paper ---------------------------------------------------------------------


def _paper() -> Paper:
    return Paper(
        arxiv_id="2106_09685",
        title="LoRA",
        authors=["Edward J. Hu"],
        published=datetime(2021, 6, 17),
        pdf_url="https://arxiv.org/pdf/2106.09685",
        content=Content(text="## Abstract\n\nLow-rank adaptation freezes W_0."),
        attributes=Attributes(
            affiliation="Microsoft", category="Finetuning", keywords=["LoRA"]
        ),
    )


@pytest.mark.parametrize("post", ["review", "summary"])
def test_a_paper_run_wires_roles_ceiling_and_shape(
    seen: Seen, monkeypatch: pytest.MonkeyPatch, post: str
) -> None:
    class FakePaperEngine:
        def __init__(self, model, settings, *, store, checkpoints) -> None:  # type: ignore[no-untyped-def]
            seen.settings.append(settings)
            seen.stores.append(store)
            assert checkpoints is store

        async def write(self, title, document, **_kwargs):  # type: ignore[no-untyped-def]
            sections = [
                ("tldr", "## TL;DR\n\n### Q\n\nA."),
                ("method", f"## 방법\n\n본문입니다.\n\n![x]({BROKEN})\n"),
                ("closing", "## 한계와 시사점\n\n끝입니다."),
            ]
            ledger = EvidenceLedger()
            ledger.items.append(_figure(document, BROKEN))
            return EvidencePaper(
                outline=None,  # type: ignore[arg-type]
                ledger=ledger,
                sections=sections,
                subtitle="부제",
                description="설명",
                notes=["engine note"],
            )

    monkeypatch.setattr(paper_engine, "EvidencePaperEngine", FakePaperEngine)
    context = _context(seen.root)
    written, notes, _blocking = asyncio.run(
        paper_writer.write_paper_with_evidence(
            context,
            _paper(),
            post=post,  # type: ignore[arg-type]
            translation_guideline=None,
            tracker=TokenUsageTracker(),
        )
    )
    if post == "review":
        cfg = context.config.explanation
        assert seen.roles["write"].invoke_id == cfg.paper_synthesis_model_id
        assert seen.ceilings == [cfg.max_cost_usd]
        assert isinstance(written, ReviewText)
        assert written.closing.startswith("## 한계와 시사점")
        assert BROKEN not in written.body
    else:
        scfg = context.config.summary
        assert seen.roles["write"].invoke_id == scfg.summary_model_id
        assert seen.ceilings == [scfg.max_cost_usd]
        assert isinstance(written, dict) and written["subtitle"] == "부제"
        assert BROKEN not in written["summary"]
    assert seen.roles["read"].invoke_id == (
        context.config.explanation.reader_model_id
        if post == "review"
        else context.config.summary.reader_model_id
    )
    assert seen.stores[0].local_dir.name == "paper-evidence"
    assert seen.stores[0].local_dir.is_relative_to(seen.root)
    assert "engine note" in notes and any(BROKEN in n for n in notes)
