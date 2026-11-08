"""The source record of gathered pages (docs/engine.md 3.1): authority,
version, date and language decided once (no network)."""

from __future__ import annotations

import json
import random

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
    release_number,
    same_release,
    url_language,
    url_version,
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
        # A versioned docs site: every section at any version is the same
        # docs (another version is labelled by its version, not here).
        (
            "https://docs.multi.dev/en/latest/serve/index.html",
            "https://docs.multi.dev/en/latest/data/overview.html",
            "https://other.multi.dev/en/latest/serve/index.html",
        ),
        # A host serving several products' docs, each under its own path
        # before the version: only the requested product's.
        (
            "https://docs.cloud.dev/alpha/latest/guide/start.html",
            "https://docs.cloud.dev/alpha/latest/api/ref.html",
            "https://docs.cloud.dev/beta/latest/guide/start.html",
        ),
        # A tenant host: the repository, not its owner's other projects.
        (
            "https://github.com/owner/proj",
            "https://github.com/owner/proj/blob/main/docs/x.md",
            "https://github.com/owner/other",
        ),
    ],
)
def test_only_the_requested_docs_site_is_official(
    requested: str, same: str, other: str
) -> None:
    sources = SourceSet.from_urls([requested])
    assert sources.official.on_host(same)
    assert not sources.official.on_host(other)


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
    for url in (
        "https://docs.example.dev/en/latest/serve/x.html",
        "https://docs.example.dev/en/latest/data/x.html",
    ):
        assert sources.classify(url).authority == "official"


@pytest.mark.parametrize(
    "url, host, scope",
    [
        ("https://d.dev/en/latest/serve/index.html", "d.dev", "en/*"),
        ("https://d.dev/alpha/v2.1/guide/", "d.dev", "alpha/*"),
        # Unversioned and no links read: the page's first path segment.
        ("https://d.dev/docs/lib/index", "d.dev", "docs"),
        ("https://d.dev/docs/lib/intro.html", "d.dev", "docs"),
        ("https://d.dev/versions/r2.10/api/index.html", "d.dev", "versions/*"),
        ("https://d.dev/", "d.dev", ""),
        ("https://github.com/Owner/Proj/tree/main/docs", "github.com", "owner/proj"),
        ("https://medium.com/@author/post-1", "medium.com", "@author"),
    ],
)
def test_the_official_scope_is_the_docs_site_or_the_tenant(
    url: str, host: str, scope: str
) -> None:
    assert official_scope(url) == (host, scope)
    assert in_scope(url, scope)


SHARE = 0.6


def _seed(url: str, links: list[str], content: list[str] | None = None) -> PageContent:
    return PageContent(
        url=url,
        title="t",
        text=TEXT,
        links=links,
        content_links=content if content is not None else links,
    )


def _sources(seed: PageContent) -> SourceSet:
    return SourceSet.from_urls([seed.url], [seed], link_share=SHARE)


def test_an_unversioned_multi_product_host_keeps_only_the_product() -> None:
    # A product page on a host that serves many products, a blog and a Q&A
    # forum: its sidebar stays in the product's docs, its header does not.
    host = "https://cloud.multi.dev"
    sidebar = [f"{host}/prod-a/docs/{p}" for p in ("start", "serve", "api", "faq")]
    sidebar += [f"{host}/prod-a/docs/guide/{p}" for p in ("one", "two", "three")]
    header = [f"{host}/prod-b/docs", f"{host}/blog", f"{host}/answers", f"{host}/"]
    seed = _seed(f"{host}/prod-a/docs/guide/overview", sidebar + header)
    sources = _sources(seed)
    assert sources.official.on_host(f"{host}/prod-a/docs/reference/cli")
    for other in (
        f"{host}/prod-b/docs/start",
        f"{host}/blog/2026/launch",
        f"{host}/answers/questions/123",
    ):
        assert sources.classify(other).authority == "third_party", other


def test_a_versioned_site_keeps_the_version_rule_whatever_the_links() -> None:
    seed = _seed(
        "https://docs.multi.dev/en/latest/serve/index.html",
        ["https://docs.multi.dev/blog/x"] * 3,
    )
    sources = _sources(seed)
    assert sources.official.on_host("https://docs.multi.dev/en/v1.2.0/data/x.html")
    assert not sources.official.on_host("https://docs.multi.dev/blog/x")


def test_a_code_host_keeps_the_tenant_rule() -> None:
    seed = _seed(
        "https://github.com/owner/proj",
        ["https://github.com/owner/other", "https://github.com/topics/x"],
    )
    sources = _sources(seed)
    assert sources.official.on_host("https://github.com/owner/proj/blob/main/a.md")
    assert not sources.official.on_host("https://github.com/owner/other")


def test_the_site_root_makes_the_whole_host_official_whatever_the_links() -> None:
    seed = _seed("https://docs.single.dev/", ["https://docs.single.dev/serve/a"])
    sources = _sources(seed)
    assert sources.official.on_host("https://docs.single.dev/train/b/c.html")


def test_a_page_without_links_scopes_to_its_first_segment() -> None:
    seed = _seed("https://cloud.multi.dev/prod-a/docs/guide/overview", [])
    sources = _sources(seed)
    assert sources.official.on_host("https://cloud.multi.dev/prod-a/pricing")
    assert not sources.official.on_host("https://cloud.multi.dev/blog/x")


def test_other_hosts_links_do_not_count_toward_the_share() -> None:
    host = "https://docs.lib.dev"
    elsewhere = [f"https://ext{i}.dev/x" for i in range(20)]
    seed = _seed(
        f"{host}/lib/guide/start",
        [f"{host}/lib/guide/{p}" for p in "abc"] + [f"{host}/news"] + elsewhere,
    )
    assert _sources(seed).official.spaces >= {("docs.lib.dev", "lib/guide")}


def test_the_scope_is_the_deepest_prefix_holding_the_share() -> None:
    """Property: over generated link sets, the scope is a prefix of the
    page's folder; at least the share of the same-host links lie under it,
    and fewer lie under the next deeper prefix (when there is one)."""
    rng = random.Random(20261108)
    words = ["a", "b", "c", "d"]
    for _ in range(300):
        depth = rng.randint(1, 4)
        folder = [rng.choice(words) for _ in range(depth)]
        url = "https://h.dev/" + "/".join(folder)
        links = []
        for _ in range(rng.randint(1, 12)):
            # Links that share a random prefix of the folder, then wander.
            keep = rng.randint(0, depth)
            tail = [rng.choice(words) for _ in range(rng.randint(0, 2))]
            links.append("https://h.dev/" + "/".join([*folder[:keep], *tail]))
        share = rng.choice([0.3, 0.5, 0.6, 0.9])
        _, scope = official_scope(url, links, link_share=share)
        got = scope.split("/")
        assert got == folder[: len(got)] and len(got) >= 1
        paths = {tuple(s for s in u.split("/")[3:] if s) for u in links}

        def under(
            n: int,
            paths: set[tuple[str, ...]] = paths,
            folder: list[str] = folder,
        ) -> int:
            return sum(1 for p in paths if p[:n] == tuple(folder[:n]))

        if under(len(got)) >= share * len(paths):
            if len(got) < depth:
                assert under(len(got) + 1) < share * len(paths)
        else:  # no prefix holds the share: the first segment
            assert len(got) == 1 and all(
                under(n) < share * len(paths) for n in range(1, depth + 1)
            )


@pytest.mark.parametrize(
    "url, value, number",
    [
        ("https://d.dev/versions/r2.10/api/index.html", "r2.10", "2.10"),
        ("https://d.dev/api_docs/r1.15.3/python", "r1.15.3", "1.15.3"),
    ],
)
def test_an_r_release_segment_is_a_release(url: str, value: str, number: str) -> None:
    found = url_version(url)
    assert found is not None and (found.kind, found.value) == ("release", value)
    assert release_number(value) == number
    assert same_release(value, number + ".1")
    # Another release line of the same docs is another version of them.
    sources = SourceSet.from_urls([url])
    other = url.replace(value, "r9.9")
    assert sources.official.on_host(other)
    assert sources.classify(other).authority == "older_docs"


@pytest.mark.parametrize("segment", ["rc1", "r2", "release", "rust"])
def test_words_starting_with_r_are_not_releases(segment: str) -> None:
    assert url_version(f"https://d.dev/docs/{segment}/x.html") is None
