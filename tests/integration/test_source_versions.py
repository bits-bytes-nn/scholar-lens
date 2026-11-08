"""One target version per run: the URL version table, the target release
and repository refs (docs/engine.md 3.1; no network)."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from scholar_lens.adapters.web.research import WebResearcher
from scholar_lens.domain.research import PageContent, ResearchCorpus
from scholar_lens.domain.sources import SourceSet
from tests.integration.test_tech_guide import _make_generator, _relevant

DOCS = "https://docs.example.dev/en/latest/index.html"
PINNED = "https://docs.example.dev/en/v0.6.0/"
ROOT = "https://docs.example.dev/"
REPO = "https://github.com/example/proj"

RELEASE_DOCS = [
    # (requested docs URLs, package release, page URL, authority, version)
    ([DOCS], None, "https://docs.example.dev/en/latest/a.html", "official", "latest"),
    ([DOCS], None, "https://docs.example.dev/en/stable/a.html", "official", "stable"),
    ([DOCS], None, "https://docs.example.dev/en/v0.6.0/a.html", "older_docs", "v0.6.0"),
    # Outside the requested docs' path scope (en/<version>): not official.
    ([DOCS], None, "https://docs.example.dev/guide/a.html", "third_party", ""),
    # A requested rolling alias stays official; another one is development.
    ([DOCS], "1.4.2", "https://docs.example.dev/en/latest/a", "official", "latest"),
    ([DOCS], "1.4.2", "https://docs.example.dev/en/nightly/a", "dev_docs", "nightly"),
    ([DOCS], "1.4.2", "https://docs.example.dev/en/v1.4/a", "official", "v1.4"),
    ([DOCS], "1.4.2", "https://docs.example.dev/en/1.3.0/a", "older_docs", "1.3.0"),
    (
        ["https://docs.example.dev/"],
        "1.4.2",
        "https://docs.example.dev/en/main/a",
        "dev_docs",
        "main",
    ),
    # A pinned docs URL is the target: stable and latest are other versions.
    ([PINNED], "1.4.2", "https://docs.example.dev/en/v0.6.0/b", "official", "v0.6.0"),
    ([PINNED], "1.4.2", "https://docs.example.dev/en/stable/b", "older_docs", "stable"),
    ([PINNED], None, "https://docs.example.dev/en/latest/b", "dev_docs", "latest"),
    # A bare major version only right after the root or a language (the
    # docs root requested: the whole host is in scope).
    ([ROOT], None, "https://docs.example.dev/v2/a", "older_docs", "v2"),
    ([ROOT], None, "https://docs.example.dev/reference/rest/v1/x", "official", ""),
    # A paper id is not a release.
    ([DOCS], None, "https://papers.example.org/papers/2106.09685", "third_party", ""),
    # Off the docs host: third-party, still with its URL version.
    ([DOCS], None, "https://blog.example.org/en/v0.6.0/x", "third_party", "v0.6.0"),
]


@pytest.mark.parametrize("urls,release,url,authority,version", RELEASE_DOCS)
def test_authority_and_version_follow_the_url_version_table(
    urls: list[str], release: str | None, url: str, authority: str, version: str
) -> None:
    record = SourceSet.from_urls(urls, package_release=release).classify(url)
    assert (record.authority, record.version) == (authority, version)


def test_the_release_a_rolling_link_landed_on_is_requested() -> None:
    seed = PageContent(
        url="https://docs.example.dev/docs/stable/index.html",
        title="t",
        text="",
        final_url="https://docs.example.dev/docs/2.14/index.html",
    )
    sources = SourceSet.from_urls([seed.url], [seed], package_release="2.15")
    assert sources.target.release == "2.15" and not sources.target.pinned
    landed = sources.classify("https://docs.example.dev/docs/2.14/x")
    assert landed.authority == "official"
    older = sources.classify("https://docs.example.dev/docs/2.0/x")
    assert older.authority == "older_docs"
    # Without a package release, the landed release is the target.
    assert SourceSet.from_urls([seed.url], [seed]).target.release == "2.14"


# ---- repository pages ----------------------------------------------------------


def _repo_set(
    release: str | None, tags: frozenset[str] | None, urls: list[str] | None = None
) -> SourceSet:
    """Docs whose pages link the project's repository, which has ``tags``."""
    sources = SourceSet.from_urls(
        urls or [DOCS], package_release=release, repo_tags=lambda _: tags
    )
    pages = [
        PageContent(
            url=f"https://docs.example.dev/en/latest/p{n}",
            title="",
            text="",
            links=[f"{REPO}/edit/main/docs/index.md"],
        )
        for n in range(3)
    ]
    return sources.with_repositories(pages)


@pytest.mark.parametrize(
    "release,tags,url,authority,version",
    [
        # At the release tag: official; on the default branch: development.
        ("1.4.2", None, f"{REPO}/blob/v1.4.2/README.md", "official_repo", "v1.4.2"),
        ("1.4.2", None, f"{REPO}/blob/main/README.md", "dev_docs", "main"),
        ("1.4.2", None, REPO, "dev_docs", ""),
        (
            "1.4.2",
            None,
            "https://raw.githubusercontent.com/example/proj/refs/tags/v1.4.2/a.md",
            "official_repo",
            "v1.4.2",
        ),
        # No release known: the newest final tag decides.
        (
            None,
            frozenset({"v2.0.0", "v1.9.0", "v2.1.0rc1"}),
            f"{REPO}/tree/v2.0.0/x",
            "official_repo",
            "v2.0.0",
        ),
        (None, frozenset({"v2.0.0"}), f"{REPO}/blob/main/x", "dev_docs", "main"),
        # A repository without tags: its default branch is the release.
        ("1.4.2", frozenset(), f"{REPO}/blob/main/x", "official_repo", "main"),
        # Nothing to compare with: the default branch is official; with the
        # tags unknown (a failed lookup), another ref may be any version.
        (None, None, f"{REPO}/blob/main/x", "official_repo", "main"),
        (None, None, REPO, "official_repo", ""),
        (None, None, f"{REPO}/blob/v0.3.0/x", "dev_docs", "v0.3.0"),
        (None, None, f"{REPO}/tree/feature-x/y", "dev_docs", "feature-x"),
        # Issues and discussions are a forum, not the project's docs.
        ("1.4.2", None, f"{REPO}/issues/12", "forum", ""),
        # Another repository is third-party.
        (
            "1.4.2",
            None,
            "https://github.com/someone/fork/blob/main/x",
            "third_party",
            "main",
        ),
    ],
)
def test_repository_pages_are_official_only_at_the_release(
    release: str | None,
    tags: frozenset[str] | None,
    url: str,
    authority: str,
    version: str,
) -> None:
    record = _repo_set(release, tags).classify(url)
    assert (record.authority, record.version) == (authority, version)


def test_the_requested_repository_ref_stays_official() -> None:
    sources = _repo_set("1.4.2", frozenset({"v1.4.2"}), urls=[REPO])
    readme = sources.classify(f"{REPO}/blob/main/README.md")
    assert readme.authority == "official_repo"
    assert sources.classify(f"{REPO}/blob/v1.0.0/README.md").authority == "dev_docs"


def test_repository_tags_are_looked_up_once_per_repository() -> None:
    calls: list[str] = []

    def lookup(repo: str) -> frozenset[str] | None:
        calls.append(repo)
        return frozenset({"v1.0.0"})

    sources = SourceSet.from_urls([DOCS], repo_tags=lookup)
    sources = sources.with_repositories(
        [PageContent(url=DOCS, title="", text="", links=[f"{REPO}/edit/main/index.md"])]
    )
    for ref in ("main", "v1.0.0", "dev"):
        sources.classify(f"{REPO}/blob/{ref}/x/y")
    assert calls == ["github.com/example/proj"]


def test_the_researcher_lists_github_tags_and_tolerates_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "example/proj" in request.url.path:
            return httpx.Response(200, json=[{"name": "v1.0.0"}, {"name": "v0.9.0"}])
        return httpx.Response(403, json={"message": "rate limited"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    researcher = WebResearcher(client=client)
    tags = researcher.repo_tags("github.com/example/proj")
    assert tags == frozenset({"v1.0.0", "v0.9.0"})
    assert researcher.repo_tags("github.com/example/other") is None
    assert researcher.repo_tags("gitlab.com/example/proj") is None
    # Only a lookup that failed is a failure (another host is no lookup).
    assert researcher.tag_lookup_failures == ["github.com/example/other"]


async def test_gather_targets_the_product_packages_latest_release() -> None:
    docs = PageContent(
        url=DOCS,
        title="Docs",
        text="Install it:\n\n```bash\npip install example-serve\n```\n",
    )
    old = PageContent(
        url="https://docs.example.dev/en/1.3.0/old.html", title="Old", text="x"
    )
    gen = _make_generator()
    gen.researcher.research = MagicMock(return_value=ResearchCorpus(pages=[docs, old]))
    gen.relevance_chain.ainvoke = _relevant()
    asked: list[str] = []

    def lookup(name: str) -> str | None:
        asked.append(name)
        return "1.4.2"

    async def gate(corpus: ResearchCorpus) -> str:
        gen._products = ("example-serve",)
        return "Example"

    gen.release_lookup = lookup
    gen._assert_relevant = gate  # type: ignore[method-assign]
    material = await gen.gather([DOCS], discover_subpages=False)
    assert asked == ["example-serve"]
    assert gen.sources.target.release == "1.4.2"
    assert [p.authority for p in material.corpus.pages] == ["official", "older_docs"]
    assert any("Target version 1.4.2" in n for n in gen.notes)
