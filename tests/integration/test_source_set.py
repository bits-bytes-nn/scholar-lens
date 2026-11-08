"""The source record of gathered pages (docs/engine.md 3.1): authority,
version, date and language decided once (no network)."""

from __future__ import annotations

import json

import pytest
from bs4 import BeautifulSoup

from scholar_lens.adapters.web.research import published_date
from scholar_lens.application.use_cases.guide import guide_documents
from scholar_lens.domain.evidence import SourceDocument
from scholar_lens.domain.research import PageContent
from scholar_lens.domain.sources import (
    SourceSet,
    in_scope,
    official_scope,
    url_language,
)

DOCS = "https://docs.example.dev/en/latest/index.html"
TEXT = "A page long enough to become a source document. " * 10


def _page(url: str, **kwargs: object) -> PageContent:
    return PageContent(url=url, title="t", text=TEXT, **kwargs)  # type: ignore[arg-type]


def test_the_record_is_written_once_and_kept() -> None:
    page = _page("https://blog.example.org/post")
    first = SourceSet.from_urls([DOCS]).record(page)
    assert first.authority == "third_party" and page.authority == "third_party"
    # A later source set that would decide otherwise does not re-decide.
    SourceSet.from_urls(["https://blog.example.org/"]).record(page)
    assert page.authority == "third_party"


def test_the_writer_reads_the_record_and_does_not_re_decide() -> None:
    repo = "https://github.com/example/proj/blob/main/docs/design.md"
    pages = [
        # Recorded as third-party at gather time: a repository URL the
        # writer once upgraded to official stays what the record says.
        _page(repo, authority="third_party"),
        _page(
            "https://github.com/example/proj/blob/v1.2.0/README.md",
            authority="official_repo",
            version="v1.2.0",
            published="2026-03-04",
            language="python",
        ),
        _page("https://docs.example.dev/en/v1.0/x.html", authority="older_docs"),
        _page("https://docs.example.dev/en/main/x.html", authority="dev_docs"),
    ]
    docs = guide_documents(pages)
    assert [(d.paper_id, d.authority, d.source_authority) for d in docs] == [
        ("src-W1", "third_party", "third_party"),
        ("src-D1", "official", "official_repo"),
        ("src-O1", "older_docs", "older_docs"),
        ("src-O2", "older_docs", "dev_docs"),
    ]
    assert (docs[1].version, docs[1].published, docs[1].language) == (
        "v1.2.0",
        "2026-03-04",
        "python",
    )


def test_source_documents_without_record_fields_still_load() -> None:
    cached = {
        "paper_id": "src-D1",
        "title": "t",
        "kind": "full_text",
        "sections": [],
        "url": "https://docs.example.dev/",
        "authority": "official",
    }
    doc = SourceDocument.model_validate_json(json.dumps(cached))
    assert (doc.source_authority, doc.version, doc.published, doc.language) == (
        "",
        "",
        "",
        "",
    )


@pytest.mark.parametrize(
    "html,expected",
    [
        (
            '<meta property="article:published_time" content="2025-04-03T10:00:00Z">',
            "2025-04-03",
        ),
        ('<meta name="date" content="2024-12-01">', "2024-12-01"),
        (
            '<script type="application/ld+json">'
            '{"@graph": [{"@type": "Article", "datePublished": "2023-07-08"}]}'
            "</script>",
            "2023-07-08",
        ),
        ('<script type="application/ld+json">not json</script>', ""),
        ("<p>no date</p>", ""),
    ],
)
def test_publication_dates_come_from_page_metadata(html: str, expected: str) -> None:
    soup = BeautifulSoup(
        f"<html><head>{html}</head><body></body></html>", "html.parser"
    )
    assert published_date(soup) == expected


def test_a_page_keeps_its_own_date_in_the_record() -> None:
    page = _page("https://blog.example.org/post", published="2025-01-02")
    assert SourceSet.from_urls([DOCS]).record(page).published == "2025-01-02"


@pytest.mark.parametrize(
    "url,language",
    [
        ("https://docs.example.dev/sdk/python/quickstart", "python"),
        ("https://docs.example.dev/sdk-for-javascript/guide", "javascript"),
        ("https://docs.example.dev/go-sdk/", "go"),
        ("https://docs.example.dev/en/latest/howto-go-faster", ""),
        ("https://docs.example.dev/en/latest/", ""),
    ],
)
def test_the_sdk_language_comes_from_the_url_path(url: str, language: str) -> None:
    assert url_language(url) == language


def test_a_source_label_names_its_record() -> None:
    from scholar_lens.domain.evidence import SourceDocument

    doc = SourceDocument.from_markdown(
        "D1", "Docs", "full_text", "## A\n\nText.", authority="official"
    )
    assert doc.label() == "official"
    recorded = doc.model_copy(
        update={
            "source_authority": "official_repo",
            "version": "v2.3.0",
            "published": "2026-05-01",
        }
    )
    assert recorded.label() == "official_repo, v2.3.0, 2026-05-01"
    assert doc.model_copy(update={"authority": ""}).label() == "third_party"


# ---- the official scope is the requested pages' path scope, on every host -----


@pytest.mark.parametrize(
    "requested, same, other",
    [
        # A multi-product docs host: one product's docs, not the others'.
        (
            "https://docs.multi.dev/en/latest/serve/index.html",
            "https://docs.multi.dev/en/2.9.0/serve/batching.html",
            "https://docs.multi.dev/en/latest/data/overview.html",
        ),
        # A host many libraries share, with no list of such hosts.
        (
            "https://hub.example.org/docs/alpha/index",
            "https://hub.example.org/docs/alpha/quicktour",
            "https://hub.example.org/docs/beta/quicktour",
        ),
        # A code host: the project, not its owner's other projects.
        (
            "https://code.example.org/owner/proj",
            "https://code.example.org/owner/proj/wiki/Home",
            "https://code.example.org/owner/other",
        ),
        # Unversioned docs: the requested section.
        (
            "https://docs.single.dev/serve/batching.html",
            "https://docs.single.dev/serve/scaling.html",
            "https://docs.single.dev/train/intro.html",
        ),
    ],
)
def test_only_the_requested_path_scope_is_official(
    requested: str, same: str, other: str
) -> None:
    sources = SourceSet.from_urls([requested])
    assert sources.official.on_host(same)
    assert sources.classify(other).authority == "third_party"


@pytest.mark.parametrize(
    "requested",
    ["https://docs.single.dev/", "https://docs.single.dev/index.html"],
)
def test_the_docs_root_makes_the_whole_host_official(requested: str) -> None:
    sources = SourceSet.from_urls([requested])
    for url in (
        "https://docs.single.dev/serve/a.html",
        "https://docs.single.dev/train/b/c.html",
    ):
        assert sources.classify(url).authority == "official"


def test_where_a_requested_page_landed_is_in_scope_too() -> None:
    seed = PageContent(
        url="https://old.example.dev/serve/",
        title="t",
        text="",
        final_url="https://docs.example.dev/en/latest/serve/index.html",
    )
    sources = SourceSet.from_urls([seed.url], [seed])
    assert (
        sources.classify("https://docs.example.dev/en/latest/serve/x.html").authority
        == "official"
    )
    assert (
        sources.classify("https://docs.example.dev/en/latest/data/x.html").authority
        == "third_party"
    )


@pytest.mark.parametrize(
    "url, scope",
    [
        ("https://d.dev/en/latest/serve/index.html", "en/*/serve"),
        ("https://d.dev/en/latest/quickstart.html", "en/*"),
        ("https://d.dev/en/latest/quickstart/", "en/*/quickstart"),
        ("https://d.dev/docs/lib/index", "docs/lib"),
        ("https://d.dev/a.html", ""),
        ("https://d.dev/", ""),
    ],
)
def test_a_requested_page_scopes_to_its_folder_below_the_version(
    url: str, scope: str
) -> None:
    assert official_scope(url) == ("d.dev", scope)
    assert in_scope(url, scope)
