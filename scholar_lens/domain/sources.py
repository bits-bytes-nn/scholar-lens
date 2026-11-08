"""The source record of every gathered page (docs/engine.md 3.1, SourceSet).

Each page of a guide's corpus gets one :class:`SourceRecord` at gather time:
its authority, version, publication date and SDK language. Everything later
(search ranking, the official-reference checks, the evidence documents, the
prompts' labels) reads the record instead of deciding again.

The rules are attributes, not sites:

* **Version** comes from a small table of URL path patterns
  (:data:`URL_VERSION_PATTERNS`): rolling aliases (``stable``; ``latest``,
  ``main``, ``nightly``) and pinned releases (``v0.6.0``, ``0.6``, ``v2``).
* **Target version**: one per run (:class:`TargetVersion`), the release the
  requested docs URL pins, else the product package's latest release on the
  package index, else the release a rolling docs link landed on.
* **Repository pages** of the project's own repositories count as official
  only at the release tag, or when the repository has no tags.
* **Demoted domain classes** (forums, social sites, aggregators) are the
  user's configured domain patterns (``tech_guide.demoted_domains``).
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast
from urllib.parse import urlparse

from scholar_lens.domain.evidence import Authority, SourceAuthority, authority_rank

VersionKind = Literal["release", "stable", "dev"]


@dataclass(frozen=True)
class VersionPattern:
    """A docs or repository version as one URL path segment: ``kind`` says
    what it means, ``max_index`` (when set) is the last segment position it
    may sit at (a bare ``v2`` deeper in a path is an API version, not a
    docs release)."""

    kind: VersionKind
    pattern: re.Pattern[str]
    max_index: int | None = None


# The URL version table: the first pattern a path segment matches wins.
URL_VERSION_PATTERNS: tuple[VersionPattern, ...] = (
    # The latest release, as the docs build names it.
    VersionPattern("stable", re.compile(r"stable", re.I)),
    # The development head: newer than any release.
    VersionPattern("dev", re.compile(r"latest|main|master|dev|nightly", re.I)),
    # A pinned release with an optional PEP 440 suffix: v0.6.0, 0.6,
    # v0.4.0.post1, 2.0rc1.
    VersionPattern(
        "release",
        re.compile(r"v?\d+(?:\.\d+)+(?:[.-]?(?:post|dev|rc|a|b)\d*)*", re.I),
    ),
    # A bare major release ("v2") right after the root or a language.
    VersionPattern("release", re.compile(r"v\d+", re.I), max_index=1),
)
# A path segment that is a paper id ("/papers/2106.09685"), not a release.
_ARXIV_ID_SEGMENT = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")
_RELEASE_NUMBER = re.compile(
    r"(?:^|[^a-z0-9])v?(\d+(?:\.\d+)*(?:[.-]?(?:post|dev|rc|a|b)\d*)*)(?![a-z0-9])"
)
_PRE_RELEASE = re.compile(r"(?:dev|rc|a|b)\d*$", re.I)

# What is official on a requested docs host is the requested docs site:
# with a docs version in the path, everything up to and including it, at
# any version ("/en/<any version>/": one product's docs on a host that
# serves many, "/<product>/<version>/", stays that product's); without one,
# the whole host, except on hosts whose first path segments name a tenant
# (a repository, an author), where it is that tenant's space.
TENANT_SEGMENTS: Mapping[str, int] = {
    "github.com": 2,
    "raw.githubusercontent.com": 2,
    "gitlab.com": 2,
    "bitbucket.org": 2,
    "huggingface.co": 2,
    "medium.com": 1,
}
# A scope segment that matches any docs version.
ANY_VERSION = "*"
# Code hosts whose URL layout names a repository and a ref: host ->
# (the host repository keys use, the path markers a ref follows).
_CODE_HOSTS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "github.com": ("github.com", ("blob", "tree", "raw", "edit", "commits")),
    "raw.githubusercontent.com": ("github.com", ()),
    "gitlab.com": ("gitlab.com", ("-/blob", "-/tree", "-/raw")),
}
# Repository sub-paths that are discussions, not the project's docs or code.
_REPO_DISCUSSION_PATHS = frozenset({"issues", "pull", "pulls", "discussions"})
# A path segment (or a word of an "sdk"-style segment) naming the SDK language.
SDK_LANGUAGES: Mapping[str, str] = {
    "python": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "java": "java",
    "kotlin": "kotlin",
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "dotnet": "dotnet",
    "csharp": "dotnet",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "cpp": "cpp",
}
# Words that mark a segment as naming an SDK ("python-sdk", "sdk-for-go").
_SDK_WORDS = frozenset({"sdk", "sdks", "client", "clients", "bindings", "api"})

# How each record authority enters the evidence engine, whose vocabulary is
# the coarser :data:`~scholar_lens.domain.evidence.Authority`.
ENGINE_AUTHORITY: Mapping[SourceAuthority, Authority] = {
    "official": "official",
    "official_repo": "official",
    "reference": "reference",
    "third_party": "third_party",
    "forum": "third_party",
    "older_docs": "older_docs",
    "dev_docs": "older_docs",
}
# Authorities whose pages decide commands, flags and defaults.
OFFICIAL_AUTHORITIES = frozenset({"official", "official_repo"})
# Authorities counted against ``max_third_party_pages``.
THIRD_PARTY_AUTHORITIES = frozenset({"third_party", "forum"})
# Authorities of the primary non-official material a guide's search slots
# keep room for (``min_secondary_pages``): papers, design docs and posts,
# reference-quality pages. Official pages (any docs version) and the
# demoted classes (forum, social, aggregator) are not among them.
SECONDARY_AUTHORITIES = frozenset({"reference", "third_party"})


def normalize_url(url: str) -> str:
    """Dedup key: scheme and host are case-insensitive, the path is not."""
    parsed = urlparse(url)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}".rstrip("/")


@dataclass(frozen=True)
class UrlVersion:
    kind: VersionKind
    value: str


def url_version(url: str) -> UrlVersion | None:
    """The first path segment the version table matches, lower-cased."""
    segments = [s for s in urlparse(url).path.split("/") if s]
    for index, segment in enumerate(segments):
        lowered = segment.lower()
        if _ARXIV_ID_SEGMENT.fullmatch(lowered):
            continue
        for row in URL_VERSION_PATTERNS:
            if row.max_index is not None and index > row.max_index:
                continue
            if row.pattern.fullmatch(lowered):
                return UrlVersion(row.kind, lowered)
    return None


def docs_version(url: str) -> str | None:
    """The docs version a URL path names (``latest``, ``stable``,
    ``v0.6.0``, ``0.6`` …), lower-cased; ``None`` when there is none."""
    found = url_version(url)
    return found.value if found else None


def release_number(value: str) -> str:
    """The release a version or tag names: ``v0.6.0`` and ``pkg-0.6.0`` →
    ``0.6.0``; "" when it names none (``main``, a commit id)."""
    found = _RELEASE_NUMBER.search(value.lower())
    return found.group(1) if found else ""


def same_release(a: str, b: str) -> bool:
    """Whether two versions name the same release: ``0.6`` docs cover
    ``0.6.3``."""
    x, y = release_number(a), release_number(b)
    if not x or not y:
        return False
    return x == y or x.startswith(y + ".") or y.startswith(x + ".")


def _release_key(value: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", release_number(value))[:4])


def newest_release(tags: Iterable[str]) -> str | None:
    """The highest final release among ``tags`` (pre-releases skipped)."""
    releases = [
        t for t in tags if (n := release_number(t)) and not _PRE_RELEASE.search(n)
    ]
    return max(releases, key=_release_key) if releases else None


def _folder(path: str) -> list[str]:
    """A path's folder segments, lower-cased: the last segment is a page
    when it has an extension (``/docs/intro.html`` → ``docs``) and a
    section otherwise (``/docs/peft/index`` keeps ``index``)."""
    segments = [seg.lower() for seg in path.split("/") if seg]
    if segments and "." in segments[-1] and not path.endswith("/"):
        segments = segments[:-1]
    return segments


def _is_version(segment: str) -> bool:
    return not _ARXIV_ID_SEGMENT.fullmatch(segment) and any(
        row.pattern.fullmatch(segment) for row in URL_VERSION_PATTERNS
    )


def official_scope(url: str) -> tuple[str, str]:
    """(host, the path scope a requested page makes official): through the
    tenant's space on a tenant host; else through the docs version when
    the path names one (the version matching any version: other versions
    of the same docs are another version, not another product); else the
    whole host (""). See :data:`TENANT_SEGMENTS`."""
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    if tenant := TENANT_SEGMENTS.get(host, 0):
        # A repository's refs (branches, tags) are judged by its own rules.
        segments = [s.lower() for s in parts.path.split("/") if s]
        return host, "/".join(segments[:tenant])
    folder = _folder(parts.path)
    found = url_version("/" + "/".join(folder) + "/") if folder else None
    if found is not None:
        return host, "/".join([*folder[: folder.index(found.value)], ANY_VERSION])
    return host, ""


def in_scope(url: str, scope: str) -> bool:
    """Whether ``url``'s path lies in ``scope`` (see :func:`official_scope`)."""
    if not scope:
        return True
    path = [seg.lower() for seg in urlparse(url).path.split("/") if seg]
    wanted = scope.split("/")
    if len(path) < len(wanted):
        return False
    return all(
        w == p or (w == ANY_VERSION and _is_version(p))
        for w, p in zip(wanted, path, strict=False)
    )


@dataclass(frozen=True)
class RepoPage:
    """A page of a code host's repository: ``repo`` is ``host/owner/name``,
    ``ref`` the branch, tag or commit ("" for the default branch), and
    ``discussion`` marks issues, pull requests and discussions."""

    repo: str
    ref: str
    discussion: bool = False


def repo_page(url: str) -> RepoPage | None:
    parts = urlparse(url)
    layout = _CODE_HOSTS.get((parts.hostname or "").lower())
    if layout is None:
        return None
    host, markers = layout
    segments = [s for s in parts.path.split("/") if s]
    if len(segments) < 2:
        return None
    repo = f"{host}/{segments[0].lower()}/{segments[1].lower().removesuffix('.git')}"
    rest = segments[2:]
    if not markers:  # raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>
        if rest[:2] == ["refs", "heads"] or rest[:2] == ["refs", "tags"]:
            rest = rest[2:]
        return RepoPage(repo, rest[0] if rest else "")
    if rest and rest[0].lower() in _REPO_DISCUSSION_PATHS:
        return RepoPage(repo, "", discussion=True)
    joined = "/".join(rest)
    for marker in markers:
        if joined.startswith(marker + "/"):
            ref = joined[len(marker) + 1 :].split("/", 1)[0]
            return RepoPage(repo, ref)
    if rest[:2] == ["releases", "tag"] and len(rest) > 2:
        return RepoPage(repo, rest[2])
    return RepoPage(repo, "")


def url_language(url: str) -> str:
    """The SDK language the URL path names ("" when none): a whole segment
    (``/docs/python/``) or a word of an SDK-style segment (``/python-sdk/``,
    ``/sdk-for-go/``)."""
    for segment in (s.lower() for s in urlparse(url).path.split("/") if s):
        if segment in SDK_LANGUAGES:
            return SDK_LANGUAGES[segment]
        words = re.split(r"[-_]+", segment)
        if len(words) > 1 and _SDK_WORDS & set(words):
            for word in words:
                if word in SDK_LANGUAGES:
                    return SDK_LANGUAGES[word]
    return ""


class _Landed(Protocol):
    url: str
    final_url: str


@dataclass(frozen=True)
class OfficialDocs:
    """The documentation the user asked for: its hosts, the docs versions the
    requested URLs name (and land on) and the requested pages' path scopes
    on each host (:func:`official_scope`)."""

    hosts: frozenset[str]
    versions: frozenset[str | None]
    spaces: frozenset[tuple[str, str]] = frozenset()

    @classmethod
    def from_urls(cls, urls: Collection[str]) -> OfficialDocs:
        return cls(
            hosts=frozenset(h for u in urls if (h := urlparse(u).hostname)),
            versions=frozenset(docs_version(u) for u in urls),
            spaces=frozenset(official_scope(u) for u in urls),
        )

    @classmethod
    def from_research(
        cls, urls: Collection[str], pages: Collection[_Landed]
    ) -> OfficialDocs:
        """Like :meth:`from_urls`, plus where the requested pages actually
        landed: a docs site that redirects (``pytorch.org/docs/stable`` →
        ``docs.pytorch.org/docs/2.14``) keeps its own pages official, and the
        pinned release a rolling or unversioned link resolves to is a
        requested version."""
        base = cls.from_urls(urls)
        requested = {normalize_url(u) for u in urls}
        hosts, versions = set(base.hosts), set(base.versions)
        spaces = set(base.spaces)
        for page in pages:
            if not page.final_url or normalize_url(page.url) not in requested:
                continue
            if host := urlparse(page.final_url).hostname:
                hosts.add(host)
                spaces.add(official_scope(page.final_url))
            asked = url_version(page.url)
            landed = docs_version(page.final_url)
            if asked is None or asked.kind != "release" or landed in base.versions:
                versions.add(landed)
        return cls(
            hosts=frozenset(hosts),
            versions=frozenset(versions),
            spaces=frozenset(spaces),
        )

    def on_host(self, url: str) -> bool:
        """On a requested docs host and inside a requested page's path scope
        there (another product's docs, a community post or a fork on the
        same host is not)."""
        host = (urlparse(url).hostname or "").lower()
        if host not in self.hosts:
            return False
        scopes = [scope for h, scope in self.spaces if h == host]
        # Hosts given without their scopes (an older record) are whole.
        return not scopes or any(in_scope(url, scope) for scope in scopes)

    def covers(self, url: str) -> bool:
        """Whether ``url`` is an official page of these docs (see
        :meth:`SourceSet.classify`)."""
        return SourceSet(self).classify(url).authority in OFFICIAL_AUTHORITIES


@dataclass(frozen=True)
class TargetVersion:
    """The one release a run's guide describes. ``pinned``: the requested
    docs URL names it (then a ``stable`` page is another version);
    ``requested``: the versions the requested URLs name or land on, whose
    pages stay official whatever they are."""

    release: str | None = None
    pinned: bool = False
    requested: frozenset[str | None] = frozenset()

    @classmethod
    def resolve(
        cls, official: OfficialDocs, urls: Collection[str], package_release: str | None
    ) -> TargetVersion:
        """The requested docs URL's pinned release, else the package
        index's latest release, else the release a rolling docs link
        landed on."""
        pins = [
            v.value
            for u in urls
            if (v := url_version(u)) is not None and v.kind == "release"
        ]
        landed = [
            v
            for v in official.versions
            if v and (found := url_version(f"/{v}")) and found.kind == "release"
        ]
        if pins:
            return cls(max(pins, key=_release_key), True, official.versions)
        release = package_release or (max(landed, key=_release_key) if landed else None)
        return cls(release, False, official.versions)

    def docs_authority(self, version: UrlVersion | None) -> SourceAuthority:
        """A requested docs host's page at ``version``."""
        if version is None or version.value in self.requested:
            return "official"
        if version.kind == "stable":
            return "older_docs" if self.pinned else "official"
        if version.kind == "dev":
            # Unknown target: a rolling alias is as current as the inputs.
            if self.release is None:
                return "official"
            return "dev_docs"
        if self.release and same_release(version.value, self.release):
            return "official"
        return "older_docs"


@dataclass(frozen=True)
class SourceRecord:
    authority: SourceAuthority
    version: str = ""
    published: str = ""
    language: str = ""


class RecordedPage(Protocol):
    url: str
    final_url: str
    user_reference: bool
    authority: str
    version: str
    published: str
    language: str


TagLookup = Callable[[str], "frozenset[str] | None"]


@dataclass(frozen=True)
class DomainClasses:
    """Configured domain classes (``forum``, ``social``, ``aggregator`` …)
    whose pages are demoted: ``classes`` maps a class name to its host
    patterns. A pattern is a host (``example.com``, also matching its
    subdomains) or a glob (``*.example.com``, ``forum.*``)."""

    classes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, classes: Mapping[str, Collection[str]]) -> DomainClasses:
        return cls(
            {
                name: tuple(p.strip().lower() for p in patterns if p.strip())
                for name, patterns in classes.items()
            }
        )

    def match(self, url: str) -> str:
        """The class of ``url``'s host ("" when none)."""
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return ""
        for name, patterns in self.classes.items():
            for pattern in patterns:
                if "*" in pattern or "?" in pattern:
                    if fnmatch.fnmatchcase(host, pattern) or (
                        pattern.startswith("*.") and host == pattern[2:]
                    ):
                        return name
                elif host == pattern or host.endswith("." + pattern):
                    return name
        return ""


def search_exclusions(domains: Iterable[str]) -> list[str]:
    """The configured domains a search backend can exclude (plain domains;
    ``*.example.com`` as ``example.com``, globs on other labels dropped)."""
    out: list[str] = []
    for raw in domains:
        domain = raw.strip().lower().removeprefix("*.")
        if domain and "*" not in domain and "?" not in domain and domain not in out:
            out.append(domain)
    return out


@dataclass
class SourceSet:
    """Decides every page's :class:`SourceRecord` (:meth:`classify`) and
    writes it on the page once (:meth:`record`)."""

    official: OfficialDocs
    target: TargetVersion = field(default_factory=TargetVersion)
    # The project's own repositories (``host/owner/name``) and the refs the
    # requested URLs name in them ("" for the default branch).
    repositories: frozenset[str] = frozenset()
    requested_refs: frozenset[tuple[str, str]] = frozenset()
    demoted: DomainClasses = field(default_factory=DomainClasses)
    repo_tags: TagLookup | None = None
    _tags: dict[str, frozenset[str] | None] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.target == TargetVersion():
            # No target given: the versions the requested docs name.
            pins = [f"/{v}" for v in self.official.versions if v]
            self.target = TargetVersion.resolve(self.official, pins, None)

    @classmethod
    def from_urls(
        cls,
        urls: Collection[str],
        pages: Collection[_Landed] = (),
        *,
        package_release: str | None = None,
        demoted: DomainClasses | None = None,
        repo_tags: TagLookup | None = None,
    ) -> SourceSet:
        """The source set of a request for ``urls`` (fetched as ``pages``)."""
        official = (
            OfficialDocs.from_research(urls, pages)
            if pages
            else OfficialDocs.from_urls(urls)
        )
        requested = [u for u in urls] + [p.final_url for p in pages if p.final_url]
        repos = {
            found
            for u in requested
            if (found := repo_page(u)) is not None and official.on_host(u)
        }
        return cls(
            official,
            TargetVersion.resolve(official, urls, package_release),
            repositories=frozenset(r.repo for r in repos),
            requested_refs=frozenset((r.repo, r.ref) for r in repos),
            demoted=demoted or DomainClasses(),
            repo_tags=repo_tags,
        )

    def with_repositories(self, pages: Iterable[_LinkedPage]) -> SourceSet:
        """This set plus the repositories the official docs pages link to
        most (their "edit on GitHub" and source links): the project's own
        repositories."""
        counts: dict[str, int] = {}
        official_pages = 0
        for page in pages:
            url = page.final_url or page.url
            if repo_page(url) or self.classify(url).authority != "official":
                continue
            official_pages += 1
            for repo in {r.repo for link in page.links if (r := repo_page(link))}:
                counts[repo] = counts.get(repo, 0) + 1
        if not counts:
            return self
        top = max(counts.values())
        # Linked from at least half as many official pages as the most linked
        # one, and from more than one when several official pages were read.
        floor = max(2 if official_pages > 2 else 1, (top + 1) // 2)
        found = {repo for repo, n in counts.items() if n >= floor}
        return SourceSet(
            self.official,
            self.target,
            repositories=self.repositories | found,
            requested_refs=self.requested_refs,
            demoted=self.demoted,
            repo_tags=self.repo_tags,
            _tags=self._tags,
        )

    def _repo_tags(self, repo: str) -> frozenset[str] | None:
        if self.repo_tags is None:
            return None
        if repo not in self._tags:
            self._tags[repo] = self.repo_tags(repo)
        return self._tags[repo]

    def _repo_authority(self, page: RepoPage) -> SourceAuthority:
        """The project's repository: official at the release tag or in a
        repository without tags, else development code; with neither a
        release nor the tags known, official on the default branch only."""
        if page.discussion:
            return "forum"
        default_branch = page.ref == "" or (
            (found := url_version(f"/{page.ref}")) is not None and found.kind == "dev"
        )
        for repo, ref in self.requested_refs:
            if repo == page.repo and (
                ref == page.ref or (default_branch and ref == "")
            ):
                return "official_repo"
        tags = self._repo_tags(page.repo)
        if tags is not None and not tags:
            return "official_repo"
        release = self.target.release or (newest_release(tags) if tags else None)
        if release is None:
            # Nothing to compare with. Unknown tags (no lookup, or it
            # failed): only the default branch is surely the project's
            # current code; another ref may be any version.
            return "official_repo" if default_branch or tags is not None else "dev_docs"
        if page.ref and same_release(page.ref, release):
            return "official_repo"
        return "dev_docs"

    def classify(
        self, url: str, *, user_reference: bool = False, published: str = ""
    ) -> SourceRecord:
        """The record of the page at ``url`` (where it landed). The one
        place authority, version and language are decided."""
        language = url_language(url)
        repo = repo_page(url)
        if repo is not None and (
            repo.repo in self.repositories or self.official.on_host(url)
        ):
            return SourceRecord(
                self._repo_authority(repo), repo.ref, published, language
            )
        version = url_version(url)
        value = version.value if version else ""
        if self.official.on_host(url):
            return SourceRecord(
                self.target.docs_authority(version), value, published, language
            )
        if user_reference:
            return SourceRecord("reference", value, published, language)
        if self.demoted.match(url):
            return SourceRecord("forum", value, published, language)
        return SourceRecord("third_party", value, published, language)

    def record(self, page: RecordedPage) -> SourceRecord:
        """Write ``page``'s record on it, once: a page that has one keeps it."""
        if page.authority:
            return SourceRecord(
                page.authority,  # type: ignore[arg-type]
                page.version,
                page.published,
                page.language,
            )
        found = self.classify(
            page.final_url or page.url,
            user_reference=page.user_reference,
            published=page.published,
        )
        page.authority = found.authority
        page.version = found.version
        page.language = found.language
        return found

    def record_all(self, pages: Iterable[RecordedPage]) -> None:
        for page in pages:
            self.record(page)


class _LinkedPage(_Landed, Protocol):
    links: list[str]


def is_official(page: RecordedPage) -> bool:
    """Whether ``page``'s record makes it official (its docs or repository)."""
    return page.authority in OFFICIAL_AUTHORITIES


def engine_authority(authority: str) -> Authority:
    """The evidence engine's authority for a record authority ("" and
    unknown values are third-party)."""
    found = ENGINE_AUTHORITY.get(cast(SourceAuthority, authority))
    return found if found is not None else "third_party"


_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def iso_date(value: str) -> str:
    """``YYYY-MM-DD`` from an ISO date or datetime ("" when there is none)."""
    found = _DATE.search(value or "")
    return found.group(0) if found else ""


def published_year(published: str) -> int:
    found = _DATE.search(published or "")
    return int(found.group(1)) if found else 0


def search_rank(record: SourceRecord, position: int) -> tuple[int, int, int, int]:
    """Sort key of a search result: authority, then recency (newer years
    first, undated pages after dated ones), then the search's own rank
    ``position``. Years, not days, so a week's difference never outweighs
    the search rank."""
    year = published_year(record.published)
    return (authority_rank(record.authority), 0 if year else 1, -year, position)
