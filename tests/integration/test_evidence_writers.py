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

from scholar_lens import bootstrap
from scholar_lens.adapters.bedrock import role_models, strands_models
from scholar_lens.adapters.bedrock.strands_models import RoleModel
from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
from scholar_lens.adapters.github import publisher
from scholar_lens.application import guide_engine
from scholar_lens.application.paper_engine import EvidencePaper
from scholar_lens.application.use_cases import guide as guide_writer
from scholar_lens.application.use_cases import paper as paper_writer
from scholar_lens.configs import Config
from scholar_lens.domain.evidence import EvidenceLedger, SourceDocument
from scholar_lens.domain.guide.models import GuideMaterial
from scholar_lens.domain.guide.sdk_index import PackageSource
from scholar_lens.domain.paper.models import Attributes, Content, Paper, ReviewText
from scholar_lens.domain.research import PageContent, ResearchCorpus
from scholar_lens.domain.sources import OfficialDocs, ReleaseUnavailable

FIGURE = "https://docs.x.dev/img/arch.png"
BROKEN = "https://docs.x.dev/img/gone.png"


@dataclass
class Seen:
    roles: dict[str, RoleModel] = field(default_factory=dict)
    ceilings: list[float | None] = field(default_factory=list)
    settings: list[Any] = field(default_factory=list)
    stores: list[Any] = field(default_factory=list)
    # The role map the engine was given (its checkpoint keys hash it).
    engine_roles: list[Any] = field(default_factory=list)
    root: Path = Path()


@pytest.fixture
def seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Seen:
    seen = Seen()

    class FakeModel:
        def __init__(self, session, roles, record) -> None:  # type: ignore[no-untyped-def]
            seen.roles = roles

    def role_model(session, region, model_id, effort, thinking):  # type: ignore[no-untyped-def]
        return RoleModel(
            invoke_id=model_id, model_label=model_id, effort=effort, thinking=thinking
        )

    def budgeted(tracker, ceiling):  # type: ignore[no-untyped-def]
        seen.ceilings.append(ceiling)
        return tracker.record

    async def unservable(urls):  # type: ignore[no-untyped-def]
        return [u for u in urls if u == BROKEN]

    monkeypatch.setattr(strands_models, "StrandsLanguageModel", FakeModel)
    monkeypatch.setattr(role_models, "role_model", role_model)
    monkeypatch.setattr(role_models, "budgeted", budgeted)
    monkeypatch.setattr(publisher, "unservable", unservable)
    seen.root = tmp_path
    return seen


def _context(root: Path) -> bootstrap.AwsRunContext:
    return bootstrap.run_context(
        Config(),
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
            roles=None,
        ) -> None:
            seen.engine_roles.append(roles)
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

    monkeypatch.setattr(guide_writer, "EvidenceGuideEngine", FakeGuideEngine)
    monkeypatch.setattr(guide_writer, "docs_packages", lambda _text, _index: {})
    context = _context(seen.root)
    guide = asyncio.run(
        guide_writer.write_guide_with_evidence(
            context,
            bootstrap.GuideToolkit(context),
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
    (engine_roles,) = seen.engine_roles
    assert engine_roles["evaluate"].model_id == cfg.evaluation_model_id
    assert engine_roles["write"].effort == cfg.thinking_effort
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


class _PinnedIndex:
    """xserve on the package index at 0.9 and 1.0 (its latest)."""

    releases = ("0.9", "1.0")

    def _at(self, release: str | None) -> str:
        if release is not None and release not in self.releases:
            raise ReleaseUnavailable(f"no xserve {release}")
        return release or "1.0"

    def extras(self, name, release=None):  # type: ignore[no-untyped-def]
        if name != "xserve":
            return None
        version = self._at(release)
        return version, frozenset({f"x{version}"})

    def source(self, name, release=None):  # type: ignore[no-untyped-def]
        if name != "xserve":
            return None
        version = self._at(release)
        return PackageSource(version, {"xserve/__init__.py": "def run():\n    pass\n"})


@pytest.mark.parametrize("releases", [("0.9", "1.0"), ("1.0",)])
def test_a_guide_run_reads_the_package_index_at_its_target_release(
    seen: Seen, monkeypatch: pytest.MonkeyPatch, releases: tuple[str, ...]
) -> None:
    given: dict[str, Any] = {}

    class Generator(FakeGenerator):
        async def gather(self, urls, **_kwargs):  # type: ignore[no-untyped-def]
            page = PageContent(
                url="https://docs.x.dev/0.9/start",
                title="Start",
                text="xserve serves models.\n\n```bash\npip install xserve[gpu]\n```\n"
                * 10,
                authority="official",
            )
            return GuideMaterial(
                topic="xserve", corpus=ResearchCorpus(pages=[page]), release="0.9"
            )

    class Engine:
        def __init__(self, *_args, package_facts, sdk=None, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            given.update(facts=package_facts, sdk=sdk)

        async def write(self, topic, documents, *, guideline):  # type: ignore[no-untyped-def]
            return guide_engine.EvidenceGuide(
                title="xserve",
                body="## xserve\n\n본문입니다.\n",
                outline=None,  # type: ignore[arg-type]
                ledger=EvidenceLedger(),
                cited=list(documents),
                notes=[],
            )

    monkeypatch.setattr(guide_writer, "EvidenceGuideEngine", Engine)
    context = _context(seen.root)
    tools = bootstrap.GuideToolkit(context)
    index = _PinnedIndex()
    index.releases = releases
    tools.packages = index  # type: ignore[assignment]
    guide = asyncio.run(
        guide_writer.write_guide_with_evidence(
            context,
            tools,
            Generator(),  # type: ignore[arg-type]
            urls=["https://docs.x.dev/0.9/start"],
            reference_urls=[],
            discover_subpages=False,
            search_queries=None,
            translation_guideline=None,
            tracker=TokenUsageTracker(),
            title_max_length=60,
        )
    )
    if "0.9" in releases:
        # The prompts and the code check name the release they read.
        assert given["facts"] == "- xserve 0.9: extras x0.9"
        assert given["sdk"] is not None and given["sdk"].version == "xserve 0.9"
        assert not any("no usable" in n for n in guide.notes)
    else:
        # Skipped with a note, never the latest release in its place.
        assert given["facts"] == "" and given["sdk"] is None
        assert sum("no usable xserve 0.9" in n for n in guide.notes) == 2


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
        def __init__(self, model, settings, *, store, checkpoints, roles) -> None:  # type: ignore[no-untyped-def]
            seen.engine_roles.append(roles)
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

    monkeypatch.setattr(paper_writer, "EvidencePaperEngine", FakePaperEngine)
    context = _context(seen.root)
    written, notes, _blocking = asyncio.run(
        paper_writer.write_paper_with_evidence(
            context,
            bootstrap.PaperToolkit(context),
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
    (engine_roles,) = seen.engine_roles
    assert engine_roles["read"].model_id == seen.roles["read"].invoke_id
    assert engine_roles["write"].model_id == seen.roles["write"].invoke_id
    assert seen.stores[0].local_dir.name == "paper-evidence"
    assert seen.stores[0].local_dir.is_relative_to(seen.root)
    assert "engine note" in notes and any(BROKEN in n for n in notes)
