"""The glue from a run to the evidence engines (src/guide_writer.py,
src/paper_writer.py): which model and fallback each role gets, the cost
ceiling, the evidence cache, and the shape of what the publisher receives.
The engines and Bedrock are replaced; no network."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
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
    # What it may link: the gathered page as requested and as served.
    assert guide.link_urls == [
        "https://docs.x.dev/guide/start",
        "https://docs.x.dev/guide/start/",
    ]


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
    else:
        scfg = context.config.summary
        assert seen.roles["write"].invoke_id == scfg.summary_model_id
        assert seen.ceilings == [scfg.max_cost_usd]
        assert isinstance(written, dict) and written["subtitle"] == "부제"
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
    assert notes == ["engine note"]


@pytest.mark.parametrize("post", ["review", "summary"])
def test_the_paper_engines_blocking_problems_reach_the_pr_body(
    seen: Seen, monkeypatch: pytest.MonkeyPatch, post: str
) -> None:
    from scholar_lens.adapters.conventions_file import load_local_conventions
    from scholar_lens.adapters.documents import paper_reader
    from scholar_lens.application import run as runtime
    from scholar_lens.domain.jobs import JobRequest

    problem = "The claim check objects to most of section S2."

    class Engine:
        def __init__(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        async def write(self, title, document, **_kwargs):  # type: ignore[no-untyped-def]
            return EvidencePaper(
                outline=None,  # type: ignore[arg-type]
                ledger=EvidenceLedger(),
                sections=[
                    ("tldr", "## TL;DR\n\n### 무엇인가요?\n\n저랭크 적응입니다."),
                    (
                        "method",
                        "## 방법\n\n가중치를 고정하고 저랭크 행렬을 학습합니다.",
                    ),
                    ("closing", "## 한계와 시사점\n\n과제를 섞기 어렵습니다."),
                ],
                subtitle="저랭크 적응",
                description="LoRA는 사전 학습 가중치를 고정합니다.",
                blocking=[problem],
            )

    monkeypatch.setattr(paper_writer, "EvidencePaperEngine", Engine)
    context = _context(seen.root)
    # The use case's return value, then the whole run.
    _written, _notes, blocking = asyncio.run(
        paper_writer.write_paper_with_evidence(
            context,
            bootstrap.PaperToolkit(context),
            _paper(),
            post=post,  # type: ignore[arg-type]
            translation_guideline=None,
            tracker=TokenUsageTracker(),
        )
    )
    assert blocking == [problem]

    context.config.explanation.engine = "evidence"
    context.config.summary.engine = "evidence"
    context.conventions = load_local_conventions()
    context.publisher_factory = lambda ctx: publisher.Publisher(
        ctx.config.resources.github,
        root_dir=seen.root,
        conventions=ctx.require_conventions(),
        fetch_image=lambda _url: b"",
    )
    paper = _paper()
    paper.attributes.controlled_tags = ["Quantization"]

    async def prepare(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return paper, None, None

    requests: list[Any] = []
    real = runtime.publish_artifact

    async def spy(context, request, *, create_pr):  # type: ignore[no-untyped-def]
        requests.append(request)
        return await real(context, request, create_pr=create_pr)

    monkeypatch.setattr(paper_reader, "prepare_paper_data", prepare)
    monkeypatch.setattr(paper_writer, "publish_artifact", spy)
    kind = "review" if post == "review" else "summarize"
    outcome = asyncio.run(
        bootstrap.run_use_case(
            context,
            JobRequest(kind=kind, source="2106.09685", create_pr=False),
            TokenUsageTracker(),
        )
    )
    assert outcome.warnings[0] == f"Blocking: {problem}"
    (request,) = requests
    body = request.pr_body_with_checklist()
    assert (
        f"### Blocking\nDo not merge before each of these is resolved:\n- [ ] {problem}"
        in body
    )


# ---- survey --------------------------------------------------------------------


def test_a_survey_write_wires_roles_and_ceiling(
    seen: Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scholar_lens.application.use_cases import survey_write

    class Built(Exception):
        pass

    class Engine:
        def __init__(self, *_args, roles, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            seen.engine_roles.append(roles)
            raise Built

    monkeypatch.setattr(survey_write, "EvidenceSurveyEngine", Engine)
    context = _context(seen.root)
    with pytest.raises(Built):
        asyncio.run(
            survey_write.write_survey_with_evidence(
                context,
                bootstrap.SurveyWriteToolkit(context),
                SimpleNamespace(papers=[]),  # type: ignore[arg-type]
                index=SimpleNamespace(records=lambda _ids: {}, openalex=None),  # type: ignore[arg-type]
                reader=SimpleNamespace(researcher=None),  # type: ignore[arg-type]
                translation_guideline="",
                tracker=TokenUsageTracker(),
                title_max_length=60,
            )
        )
    cfg = context.config.survey
    assert seen.roles["write"].invoke_id == cfg.writing_model_id
    assert seen.roles["read"].invoke_id == cfg.card_model_id
    assert seen.ceilings == [cfg.write_max_cost_usd]


# ---- the guide's identifier check -----------------------------------------------

_OFFICIAL = (
    "xserve serves models.\n\n```bash\npip install xserve[gpu]\n"
    "xserve serve --port 8000\n```\n\n```python\nfrom xserve import run\n"
    'run("m", port=8000)\n```\n'
) * 4

# Each line breaks one rule: an install extra the index lacks, a flag no page
# mentions, an import path and a keyword the package does not have.
_WRONG_BODY = (
    "## xserve\n\n```bash\npip install xserve[turbo]\n"
    "xserve serve --port 8000 --warp-speed\n```\n\n```python\n"
    "from xserve import run\nfrom xserve.engines import Fast\n"
    'run("m", colour=1)\n```\n'
)


class _ServeIndex:
    """xserve 1.0 on the package index, its code read when ``code``."""

    def __init__(self, code: bool) -> None:
        self.code = code

    def extras(self, name, release=None):  # type: ignore[no-untyped-def]
        return ("1.0", frozenset({"gpu"})) if name == "xserve" else None

    def source(self, name, release=None):  # type: ignore[no-untyped-def]
        if name != "xserve" or not self.code:
            return None
        return PackageSource(
            "1.0", {"xserve/__init__.py": "def run(model, port=8000):\n    pass\n"}
        )


def _checked_guide(
    seen: Seen, monkeypatch: pytest.MonkeyPatch, *, code: bool
) -> tuple[list[str], list[str]]:
    """The engine's section hook problems and the final notes of a guide
    whose body is ``_WRONG_BODY``, the identifier check built by the run."""
    hook: list[str] = []

    class Generator(FakeGenerator):
        async def gather(self, urls, **_kwargs):  # type: ignore[no-untyped-def]
            page = PageContent(
                url="https://docs.x.dev/guide/start",
                title="Start",
                text=_OFFICIAL,
                authority="official",
            )
            return GuideMaterial(topic="xserve", corpus=ResearchCorpus(pages=[page]))

    class Engine(guide_engine.EvidenceGuideEngine):
        async def write(self, topic, documents, *, guideline):  # type: ignore[no-untyped-def]
            ctx = SimpleNamespace(outline=SimpleNamespace(code_language="python"))
            hook.extend(
                p.text
                for p in self.extra_problems(ctx, _WRONG_BODY)  # type: ignore[arg-type]
                if p.kind == "unknown_flags"
            )
            return guide_engine.EvidenceGuide(
                title="xserve",
                body=_WRONG_BODY,
                outline=None,  # type: ignore[arg-type]
                ledger=EvidenceLedger(),
                cited=list(documents),
                notes=[],
            )

    monkeypatch.setattr(guide_writer, "EvidenceGuideEngine", Engine)
    context = _context(seen.root)
    tools = bootstrap.GuideToolkit(context)
    tools.packages = _ServeIndex(code)  # type: ignore[assignment]
    guide = asyncio.run(
        guide_writer.write_guide_with_evidence(
            context,
            tools,
            Generator(),  # type: ignore[arg-type]
            urls=["https://docs.x.dev/guide/start"],
            reference_urls=[],
            discover_subpages=False,
            search_queries=None,
            translation_guideline=None,
            tracker=TokenUsageTracker(),
            title_max_length=60,
        )
    )
    (note,) = [n for n in guide.notes if n.startswith("Flags/metrics")]
    return hook, [note]


def test_the_guide_check_reports_what_the_package_code_lacks(
    seen: Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook, notes = _checked_guide(seen, monkeypatch, code=True)
    for problem in (
        "`run()` takes no keyword 'colour'",
        "module `xserve.engines` does not exist",
        "--warp-speed",
        "xserve[turbo]",
    ):
        assert any(problem in p for p in hook), problem
        assert problem in notes[0], problem
    # What the package's code defines is not reported.
    assert not any("[port]" in p or "[--port]" in p for p in hook)


def test_the_guide_check_without_package_code_reports_unseen_imports(
    seen: Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook, notes = _checked_guide(seen, monkeypatch, code=False)
    for problem in ("[xserve.engines]", "[--warp-speed]", "[xserve[turbo]]"):
        assert any(problem in p for p in hook), problem
    for item in ("xserve.engines", "--warp-speed", "xserve[turbo]"):
        assert item in notes[0], item
    # The import path the official pages show is not reported.
    assert not any("[xserve]" in p for p in hook)


def test_figures_that_cannot_be_downloaded_leave_the_paper_before_writing(
    seen: Seen,
) -> None:
    """Both engines then write only about figures the post can show."""
    from scholar_lens.domain.paper.models import Figure

    paper = _paper()
    text = (
        f'{paper.content.text}\n[Image: alt="a", src="{FIGURE}"]\n'
        f'[Image: alt="b", src="{BROKEN}"]\n'
    )
    paper = paper.model_copy(
        update={
            "figures": [
                Figure(figure_id="1", path=FIGURE),
                Figure(figure_id="2", path=BROKEN),
            ],
            "content": paper.content.model_copy(update={"text": text}),
        }
    )
    context = _context(seen.root)
    kept, notes = asyncio.run(
        paper_writer.servable_paper(paper, bootstrap.PaperToolkit(context).figures)
    )
    assert [f.path for f in kept.figures] == [FIGURE]
    # Neither the figure list nor the text's markers keep it (the sections
    # engine reads its figures from the markers).
    assert FIGURE in kept.content.text and BROKEN not in kept.content.text
    document, _ = paper_writer.paper_documents(kept)
    assert [f.url for f in document.figures] == [FIGURE]
    assert notes == [f"Figures that cannot be downloaded were left out: {BROKEN}"]
