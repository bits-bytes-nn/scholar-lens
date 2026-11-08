"""Artifact-agnostic publishing: save markdown, upload to S3, open a blog PR.

Reviews, summaries and technical guides all produce a Markdown document plus an
optional directory of figures. This module centralises the "ship it" steps so
every artifact type publishes identically:

* localize the paper's figures: arXiv and PDF-extracted figures are stored
  flat as ``assets/images/<slug>-<n>.<ext>`` and referenced by absolute path,
  the blog's convention (arXiv links rot when versions change and get no
  layout dimensions); any other image is someone else's and stays a link to
  its source,
* write the Markdown to a local file and upload the post + images to S3, and
* commit the post + images to a deterministic branch of the configured
  github.io repository and open (or update) its PR (:mod:`.blog_sync`).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

import httpx
from github import Auth, Github, GithubRetry

from scholar_lens.adapters.aws.storage import S3Handler, S3OperationError
from scholar_lens.adapters.documents.images import (
    is_ar5iv_placeholder,
    is_svg,
    raster_to_png,
    source_figure,
    svg_to_png,
)
from scholar_lens.adapters.github.blog_sync import BlogBranchSync, GithubSettings
from scholar_lens.adapters.http.pinned_transport import PinnedHTTPTransport
from scholar_lens.adapters.http.rate_limit import (
    ARXIV_RATE_LIMITER,
    arxiv_retry_after_seconds,
)
from scholar_lens.adapters.http.url_guard import UnsafeUrlError, assert_url_is_public
from scholar_lens.domain.constants import AppConstants, EnvVars, LocalPaths, S3Paths
from scholar_lens.domain.conventions import (
    BlogPost,
    PostConventions,
    load_local_conventions,
)
from scholar_lens.domain.ids import ARXIV_ID_BODY, is_arxiv_host
from scholar_lens.domain.markdown.front_matter import split_front_matter_text
from scholar_lens.domain.markdown.lint import (
    drop_empty_headings,
    publish_body,
    writing_notes,
)
from scholar_lens.domain.markdown.math import IMAGE, segments
from scholar_lens.domain.post_names import post_slug, same_post
from scholar_lens.domain.prompt_safety import strip_defanging
from scholar_lens.domain.publishing import (
    BLOG_IMAGES_DIR,
    REJECTED_IMAGES_DIRNAME,
    REJECTED_SUFFIX,
    VIOLATIONS_OPENING,
    PublishError,
    PublishRequest,
    SavedPublishRequest,
    is_post_file,
    request_name,
)
from scholar_lens.domain.validation import PostValidationError, validate_post

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from github.Repository import Repository

_IMAGE_DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
# Whole-download bound: the timeout above is per read, so a server dripping
# bytes could otherwise stall the publish for as long as it likes.
_IMAGE_DOWNLOAD_DEADLINE_SECONDS = 60.0
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
# A paper's arXiv source (LaTeX + figures), read when a figure is ar5iv's
# placeholder; arXiv caps submissions at 50 MB.
_SOURCE_DOWNLOAD_DEADLINE_SECONDS = 180.0
_MAX_SOURCE_BYTES = 100 * 1024 * 1024
# Largest file stored as it is: every image is committed to the public blog
# repository (and again on each rerun that changes it). A larger PNG or JPEG
# is scaled to _SHRUNK_SIDE; a GIF (its animation would be lost) is not.
_MAX_STORED_BYTES = 5 * 1024 * 1024
_SHRUNK_SIDE = 2048
# Sent on every download: names the tool to the sites it fetches from.
_USER_AGENT = "scholar-lens (+https://github.com/bits-bytes-nn/scholar-lens)"
# Formats stored as they are, by magic bytes -> stored extension (a GIF keeps
# its animation).
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
)
# The figure path inside an arXiv HTML figure URL: /html/<id>v<N>/<path>.
_ARXIV_HTML_FIGURE = re.compile(rf"^/html/({ARXIV_ID_BODY})/(.+)$")

# GitHub API client limits: per-request timeout (seconds) and a small retry
# budget (GithubRetry also waits out secondary rate limits).
_GITHUB_TIMEOUT_SECONDS = 30
_GITHUB_MAX_RETRIES = 3


class PostAssetError(Exception):
    """An image the post references could not be localized; nothing is
    uploaded or proposed. The post is kept as a rejected draft."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        # Where the rejected post was kept (set by the publisher).
        self.rejected_path: str | None = None
        self.rejected_s3_uri: str | None = None


def image_format(data: bytes) -> str | None:
    """Stored extension for a PNG/JPEG/GIF payload; None for anything else."""
    for signature, extension in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return extension
    return None


def download_image(url: str) -> bytes:
    """Fetch an image over the SSRF-pinned transport, with timeouts, a total
    deadline and a size cap. Every redirect hop is re-validated by the
    transport."""
    return _download(url, _MAX_IMAGE_BYTES, _IMAGE_DOWNLOAD_DEADLINE_SECONDS)


def download_source(url: str) -> bytes:
    """Fetch an arXiv source archive (``/e-print/<id>``), paced by the
    process-wide arXiv rate limiter."""
    ARXIV_RATE_LIMITER.acquire()
    try:
        return _download(url, _MAX_SOURCE_BYTES, _SOURCE_DOWNLOAD_DEADLINE_SECONDS)
    except httpx.HTTPStatusError as e:
        ARXIV_RATE_LIMITER.penalize(arxiv_retry_after_seconds(e))
        raise


def _download(url: str, max_bytes: int, deadline_seconds: float) -> bytes:
    assert_url_is_public(url)
    deadline = _clock() + deadline_seconds
    with httpx.Client(
        transport=PinnedHTTPTransport(),
        timeout=_IMAGE_DOWNLOAD_TIMEOUT,
        follow_redirects=True,
        max_redirects=5,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"download exceeds {max_bytes} bytes")
                if _clock() > deadline:
                    raise ValueError(f"download took over {deadline_seconds:g} s")
                chunks.append(chunk)
    return b"".join(chunks)


_clock = time.monotonic  # replaced in tests


def _is_paper_figure(src: str) -> bool:
    """A figure of the paper itself (cropped from its PDF, or on arXiv): the
    only images the blog stores. Any other image (docs sites, blogs) is
    copyrighted by its owner and stays a link to it."""
    return not src.startswith(("http://", "https://")) or is_arxiv_host(src)


_HTML_IMG_SRC = re.compile(
    r"""(?P<lead><img\b[^>]*?\bsrc=["'])(?P<url>https?://[^"'\s]+)""", re.IGNORECASE
)
_HTML_HREF = re.compile(
    r"""(?P<lead><a\b[^>]*?\bhref=["'])(?P<url>https?://[^"'\s]+)""", re.IGNORECASE
)


class ImageLocalizer:
    """Rewrite the paper's figures in a post to the blog's flat
    ``/assets/images`` layout, writing the image files into ``staging_dir``.

    * arXiv URLs are downloaded over HTTPS (an ``http://`` URL is upgraded:
      the Batch job reaches port 443 only; public hosts only, every redirect
      hop checked),
    * local figures (bare names or scratch paths) are read from
      ``figures_dir``,
    * any other remote image, paths already under ``/assets/`` and ``data:``
      URLs are left alone (the validator flags the remote ones).

    PNG, JPEG and GIF are stored as they are; an SVG is rasterized and any
    other raster format converted to PNG. A file over
    :data:`_MAX_STORED_BYTES` is scaled down (a GIF cannot be).

    An arXiv figure that is ar5iv's "NO IMAGE AVAILABLE" placeholder is taken
    from the paper's arXiv source instead.

    Images are named ``<slug>-<n>.<ext>`` in order of first appearance; the
    same source reused in the post maps to the same file. A figure that
    cannot be stored raises :class:`PostAssetError`. An image already staged
    that the post still shows by its blog path (a saved post published
    again) is kept, and new images are numbered past it.
    """

    def __init__(
        self,
        *,
        slug: str,
        figures_dir: Path,
        staging_dir: Path,
        fetch: Callable[[str], bytes] = download_image,
        fetch_source: Callable[[str], bytes] = download_source,
    ) -> None:
        self.slug = slug
        self.figures_dir = figures_dir
        self.staging_dir = staging_dir
        self.fetch = fetch
        self.fetch_source = fetch_source

    def localize(self, markdown: str) -> str:
        """The post with the paper's figures stored."""
        kept: dict[str, bytes] = {}
        if self.staging_dir.exists():
            # This post's own scratch folder: drop a previous run's images,
            # except those the post already shows by their blog path.
            kept = {
                path.name: path.read_bytes()
                for path in self.staging_dir.iterdir()
                if path.is_file() and f"/{BLOG_IMAGES_DIR}/{path.name}" in markdown
            }
            shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True)
        for name, data in kept.items():
            (self.staging_dir / name).write_bytes(data)
        taken = {Path(name).stem for name in kept}
        numbers = itertools.count(1)
        hosted: dict[str, str] = {}
        failures: list[str] = []
        # arXiv id -> its source archive, or why it could not be fetched.
        sources: dict[str, bytes | PostAssetError] = {}

        code: list[tuple[int, int]] = []

        def in_code(pos: int) -> bool:
            return any(start <= pos < end for start, end in code)

        def substitute(
            pattern: re.Pattern[str], repl: Callable[[re.Match[str]], str], text: str
        ) -> str:
            # Each pass shortens the text before later matches (a long URL
            # becomes a short path): find the code spans afresh every time.
            code[:] = [
                (seg.start, seg.end)
                for seg in segments(text)
                if seg.kind in ("fence", "icode")
            ]
            return pattern.sub(repl, text)

        def store(src: str) -> str | None:
            """The blog path of ``src`` once stored, or None when left as-is."""
            if src in hosted:
                return hosted[src]
            if src.startswith(("/assets/", "data:")) or not _is_paper_figure(src):
                return None  # left as-is by policy
            try:
                data, extension = self._image(src, sources)
            except PostAssetError as e:
                failures.append(str(e))
                return None
            stem = next(
                stem for n in numbers if (stem := f"{self.slug}-{n}") not in taken
            )
            name = f"{stem}.{extension}"
            (self.staging_dir / name).write_bytes(data)
            hosted[src] = f"/{BLOG_IMAGES_DIR}/{name}"
            return hosted[src]

        def repl(match: re.Match) -> str:
            if in_code(match.start()):
                return match.group(0)  # an image example inside code
            alt_text, src = match.group(1), match.group(2).strip()
            title = match.group(3) or ""
            path = store(src)
            return match.group(0) if path is None else f"![{alt_text}]({path}{title})"

        def html_src(match: re.Match) -> str:
            if in_code(match.start()):
                return match.group(0)
            path = store(match.group("url"))
            return match.group(0) if path is None else match.group("lead") + path

        markdown = substitute(IMAGE, repl, markdown)
        # An HTML <img> (a hand-written figure or lightbox) stores the same way,
        # and a lightbox link pointing at that image follows it.
        markdown = substitute(_HTML_IMG_SRC, html_src, markdown)
        markdown = substitute(
            _HTML_HREF,
            lambda m: (
                m.group("lead") + hosted[m.group("url")]
                if m.group("url") in hosted and not in_code(m.start())
                else m.group(0)
            ),
            markdown,
        )
        if failures:
            raise PostAssetError("; ".join(failures))
        return markdown

    def _image(
        self, src: str, sources: dict[str, bytes | PostAssetError]
    ) -> tuple[bytes, str]:
        """The stored bytes and extension for ``src``; raises
        :class:`PostAssetError` saying why it cannot be stored."""
        data = self._load(src)
        if is_ar5iv_placeholder(data):
            data = self._source_figure(src, sources)
        if is_svg(data):
            try:
                data = svg_to_png(data)  # stored as a raster, never as markup
            except (ValueError, UnicodeDecodeError) as e:
                raise PostAssetError(
                    f"'{src}' is an SVG that could not be rendered: {e}"
                ) from e
        extension = image_format(data)
        try:
            if extension is None:
                data, extension = raster_to_png(data), "png"
            if len(data) > _MAX_STORED_BYTES and extension != "gif":
                data, extension = raster_to_png(data, _SHRUNK_SIDE), "png"
        except ValueError as e:
            raise PostAssetError(f"'{src}' is not an image: {e}") from e
        if len(data) > _MAX_STORED_BYTES:
            raise PostAssetError(
                f"'{src}' is over {_MAX_STORED_BYTES} bytes "
                f"{'(an animated GIF)' if extension == 'gif' else 'even when scaled down'}"
            )
        return data, extension

    def _load(self, src: str) -> bytes:
        if src.startswith(("http://", "https://")):
            url = "https://" + src.split("://", 1)[1]
            try:
                return self.fetch(url)
            except (httpx.HTTPError, UnsafeUrlError, ValueError) as e:
                raise PostAssetError(f"could not download '{url}': {e}") from e
        local = self.figures_dir / Path(src).name
        if not local.is_file():
            raise PostAssetError(f"figure '{src}' not found in {self.figures_dir}")
        return local.read_bytes()

    def _source_figure(
        self, src: str, sources: dict[str, bytes | PostAssetError]
    ) -> bytes:
        """The figure ``src`` (ar5iv's placeholder) from the paper's arXiv
        source."""
        match = _ARXIV_HTML_FIGURE.match(urlsplit(src).path)
        if not is_arxiv_host(src) or match is None:
            raise PostAssetError(f"'{src}' is ar5iv's 'NO IMAGE AVAILABLE' image")
        arxiv_id, figure_path = match.group(1), unquote(match.group(2))
        if arxiv_id not in sources:
            url = f"{AppConstants.External.ARXIV_E_PRINT.value}/{arxiv_id}"
            try:
                sources[arxiv_id] = self.fetch_source(url)
            except (httpx.HTTPError, UnsafeUrlError, ValueError) as e:
                sources[arxiv_id] = PostAssetError(
                    f"'{src}' is a placeholder image and the arXiv source "
                    f"could not be downloaded: {e}"
                )
        archive = sources[arxiv_id]
        if isinstance(archive, PostAssetError):
            raise archive
        try:
            return source_figure(archive, figure_path)
        except ValueError as e:
            raise PostAssetError(
                f"'{src}' is a placeholder image and the arXiv source has no "
                f"usable figure for it: {e}"
            ) from e


class Publisher:
    """Publishes artifacts to S3 and the configured blog repository."""

    def __init__(
        self,
        github_config: GithubSettings,
        *,
        root_dir: Path,
        s3_handler: S3Handler | None = None,
        s3_bucket_name: str | None = None,
        s3_prefix: str = "",
        conventions: PostConventions | None = None,
        site_posts: list[BlogPost] | None = None,
        fetch_image: Callable[[str], bytes] = download_image,
        fetch_source: Callable[[str], bytes] = download_source,
    ) -> None:
        self.github_config = github_config
        self.conventions = conventions or load_local_conventions()
        # The blog's posts a new one must not clash with (descriptions,
        # translation ids); none when the blog could not be listed.
        self.site_posts = site_posts or []
        self.root_dir = root_dir
        self.s3_handler = s3_handler
        self.s3_bucket_name = s3_bucket_name
        self.s3_prefix = s3_prefix
        self.fetch_image = fetch_image
        self.fetch_source = fetch_source

    async def publish(self, request: PublishRequest) -> tuple[str | None, Path]:
        """Normalize, localize images, validate, save and upload the post.

        Returns ``(s3_url, local_markdown_path)``. Raises
        :class:`PostAssetError` (or the validator's error) when the post
        cannot be published as-is; it is then kept only as a rejected draft
        (work dir, and S3 when configured).
        """
        slug = request.slug
        # The post filename is prefixed with the publish date (today); the paper's
        # own date lives in the front-matter `date` field, set by the caller. A
        # post that replaces a published one keeps its file name: the name it
        # is validated with is the one the blog gets.
        file_name = (
            Path(request.published_path).stem
            if request.published_path
            else request.post_name or f"{datetime.now(UTC).strftime('%Y-%m-%d')}-{slug}"
        )
        # Saved with the request: a later publish validates the same name.
        request.post_name = file_name
        # Body-only passes: the front matter is already YAML-escaped and must
        # not be math-normalized or HTML-escaped.
        # Defanged prompt tags a model copied from its input get their plain
        # "<" back everywhere (titles too; harmless inside YAML strings).
        front_matter, body = split_front_matter_text(strip_defanging(request.markdown))
        # A fence left open at the end is closed (sections are closed one by
        # one already); then the blog math rules ($$ only, no \_ in math,
        # blank lines around display blocks) and the other kramdown/MathJax
        # auto-fixes.
        body = publish_body(body)
        markdown = front_matter + body
        localizer = ImageLocalizer(
            slug=slug,
            figures_dir=request.work_dir / LocalPaths.FIGURES_DIR.value,
            staging_dir=request.staged_images_dir,
            fetch=self.fetch_image,
            fetch_source=self.fetch_source,
        )
        try:
            markdown = await asyncio.to_thread(localizer.localize, markdown)
            # An image the localizer dropped can leave its heading empty.
            markdown = drop_empty_headings(markdown)
        except PostAssetError as e:
            # The finished text must not be lost with its figures.
            rejected = self._save_rejected(markdown, request, [f"- image: {e}"])
            logger.error("Post blocked by its images; kept at '%s'", rejected)
            e.rejected_path = str(rejected)
            await self._keep_rejected_copy(e, request)
            raise
        try:
            request.validation_warnings = [
                *request.review_notes,
                # Rule slips the generators could not fix, on the final text.
                *writing_notes(body),
                *self._validate(markdown, request, f"{file_name}.md"),
            ]
        except PostValidationError as e:
            await self._keep_rejected_copy(e, request)
            raise

        markdown_path = request.work_dir / f"{file_name}.md"
        await asyncio.to_thread(markdown_path.write_text, markdown, encoding="utf-8")
        await asyncio.to_thread(self._save_request, request, markdown_path)

        s3_url = await self._upload_to_s3(request, markdown_path)
        return s3_url, markdown_path

    def find_published(
        self, slug: str, branch_id: str, category: str
    ) -> BlogPost | None:
        """The blog's post a new one with ``slug``, from the artifact
        ``branch_id``, replaces: the post with that slug (older slug rules
        included, see :func:`same_post`) whose ``categories[0]`` is
        ``category`` (a guide titled like a review does not take it over),
        else the one this tool wrote on the artifact's branch and a merged
        pull request published (a rerun under a new title). The new post then
        keeps its file name, URL, date, topic and translation pairing.

        Without a blog listing nothing is known to be published, and without
        a GitHub token there are no pull requests to read: None.

        Raises :class:`PublishError` when the blog's pull requests cannot be
        read.
        """
        posts = {post.path: post for post in self.site_posts if is_post_file(post.path)}
        by_slug = sorted(
            path
            for path, post in posts.items()
            if same_post(post_slug(Path(path).stem), slug)
            and post.categories[:1] == [category]
        )
        if by_slug:
            return posts[by_slug[0]]
        if not posts or not os.getenv(EnvVars.GITHUB_TOKEN.value):
            return None
        published = self._with_repo(
            lambda repo: BlogBranchSync(repo, self.github_config).merged_posts(
                branch_id
            )
        )
        # A translation ("…-en.md") on the branch is not the post this run writes.
        return next(
            (
                posts[path]
                for path in published
                if path in posts and not Path(path).stem.endswith("-en")
            ),
            None,
        )

    def _validate(
        self, markdown: str, request: PublishRequest, file_name: str
    ) -> list[str]:
        """Check the finished post against the blog conventions.

        Errors stop publishing (nothing saved, uploaded or proposed); the post
        is kept in the work dir as ``*.rejected.md`` with the violations so
        it can be inspected. Warnings are returned for the PR checklist.
        """
        violations = validate_post(
            markdown,
            self.conventions,
            request.kind,
            file_name=file_name,
            site_posts=self.site_posts,
        )
        if blocking := [v for v in violations if v.level == "error"]:
            rejected = self._save_rejected(
                markdown, request, [f"- {v}" for v in violations]
            )
            logger.error(
                "Post rejected with %d error(s); kept at '%s'", len(blocking), rejected
            )
            error = PostValidationError(violations)
            error.rejected_path = str(rejected)
            raise error
        return [str(v) for v in violations]

    @staticmethod
    def _save_request(request: PublishRequest, markdown_path: Path) -> Path:
        """Save ``request`` without its text next to the post it published
        or rejected (``<name>.request.json``): ``scholar-lens publish``
        reads it back."""
        path = markdown_path.with_name(request_name(markdown_path.name))
        path.write_text(
            SavedPublishRequest.of(request).model_dump_json(indent=2), encoding="utf-8"
        )
        return path

    @staticmethod
    def _save_rejected(
        markdown: str, request: PublishRequest, findings: list[str]
    ) -> Path:
        """Keep a post that is not published as ``<slug>.rejected.md`` in the
        work dir, with why and the generators' notes at its end, and its
        publish request next to it."""
        rejected = request.work_dir / f"{request.slug}{REJECTED_SUFFIX}"
        report = "\n".join(
            [
                *(f"- blocking: {b}" for b in request.blocking),
                *findings,
                *(f"- note: {n}" for n in request.review_notes),
            ]
        )
        rejected.write_text(
            f"{markdown}\n\n{VIOLATIONS_OPENING}{report}\n-->\n", encoding="utf-8"
        )
        Publisher._save_request(request, rejected)
        return rejected

    async def _keep_rejected_copy(
        self, error: PostValidationError | PostAssetError, request: PublishRequest
    ) -> None:
        """In Batch the work dir is container scratch: copy the rejected post
        to S3, its publish request next to it and its localized images
        under ``rejected/blog-images/<slug>/`` (what ``publish`` needs). A
        failed copy must not hide why the post was rejected."""
        if not (error.rejected_path and self.s3_handler):
            return
        key = f"{self.s3_prefix}/rejected".lstrip("/")
        path = Path(error.rejected_path)
        try:
            await self.s3_handler.upload_file_async(path, key)
            error.rejected_s3_uri = f"s3://{self.s3_bucket_name}/{key}/{path.name}"
            await self._copy_request(key, path)
            staged = request.staged_images_dir
            if staged.is_dir() and any(staged.iterdir()):
                await asyncio.to_thread(
                    self.s3_handler.upload_directory,
                    staged,
                    f"{key}/{REJECTED_IMAGES_DIRNAME}/{request.slug}",
                )
        except Exception as upload_error:  # noqa: BLE001
            logger.warning("Could not copy the rejected post: %s", upload_error)

    async def _copy_request(self, key: str, markdown_path: Path) -> None:
        """The saved publish request of ``markdown_path`` next to its S3
        copy under ``key``; a failed copy is logged (the post is kept)."""
        assert self.s3_handler is not None
        path = markdown_path.with_name(request_name(markdown_path.name))
        try:
            await asyncio.to_thread(
                self.s3_handler.put_text,
                f"{key}/{path.name}",
                path.read_text(encoding="utf-8"),
            )
        except (OSError, S3OperationError) as e:
            logger.warning("Could not copy the publish request %s: %s", path.name, e)

    async def _upload_to_s3(
        self, request: PublishRequest, markdown_path: Path
    ) -> str | None:
        if not self.s3_handler:
            return None
        prefix = self.s3_prefix or ""
        posts_key = f"{prefix}/{S3Paths.POSTS.value}".lstrip("/")
        images_key = f"{prefix}/{BLOG_IMAGES_DIR}".lstrip("/")
        await self.s3_handler.upload_file_async(markdown_path, posts_key)
        await self._copy_request(posts_key, markdown_path)

        staged = request.staged_images_dir
        if staged.is_dir() and any(staged.iterdir()):
            await asyncio.to_thread(
                self.s3_handler.upload_directory, staged, images_key
            )
        return f"s3://{self.s3_bucket_name}/{posts_key}/{markdown_path.name}"

    async def create_pull_request(
        self, request: PublishRequest, markdown_path: Path
    ) -> str:
        """Commit the post (+ images) to ``<branch_prefix>/<branch_id>`` and
        return the URL of its open PR, creating or updating the PR as needed.

        Re-running the same artifact updates the same branch and PR. Raises
        :class:`PublishError` on any failure; the token never appears in it.
        """
        # PyGithub is blocking HTTP; keep it off the event loop.
        return await asyncio.to_thread(
            self._with_repo,
            lambda repo: BlogBranchSync(repo, self.github_config).sync(
                request, markdown_path
            ),
        )

    def _with_repo[T](self, action: Callable[[Repository], T]) -> T:
        """``action`` on the blog repository; any failure is a
        :class:`PublishError` without the token in it."""
        repo_name = self.github_config.repo_name
        if not repo_name:
            raise PublishError("resources.github.repo_name is not configured.")
        token = os.getenv(EnvVars.GITHUB_TOKEN.value)
        if not token:
            raise PublishError(f"{EnvVars.GITHUB_TOKEN.value} is not set.")
        client = Github(
            auth=Auth.Token(token),
            timeout=_GITHUB_TIMEOUT_SECONDS,
            retry=GithubRetry(total=_GITHUB_MAX_RETRIES),
        )
        try:
            return action(client.get_repo(repo_name))
        except PublishError:
            raise
        except Exception as e:
            raise PublishError(str(self._redact_token(e, token))) from None
        finally:
            client.close()

    @staticmethod
    def _redact_token(error: Exception, token: str | None) -> Exception:
        """Return ``error`` with the token scrubbed from its text.

        A plain RuntimeError is returned rather than re-constructing the
        original class (structured exceptions such as GithubException take
        constructor args); the original class name is kept in the message.
        """
        if not token:
            return error
        scrubbed = str(error).replace(token, "***")
        return RuntimeError(f"{type(error).__name__}: {scrubbed}")


def _downloadable(url: str) -> bool:
    try:
        download_image(url)
    except Exception as e:  # noqa: BLE001 - any failure means it cannot be published
        logger.warning("Figure %s cannot be downloaded: %s", url, e)
        return False
    return True


async def unservable(urls: list[str]) -> list[str]:
    """The figure URLs that cannot be downloaded (arXiv HTML lists some it
    does not serve): an image that fails to download fails the publish
    step."""
    ok = await asyncio.gather(*(asyncio.to_thread(_downloadable, u) for u in urls))
    return [u for u, good in zip(urls, ok, strict=True) if not good]


class FigureDownloads:
    """The :class:`~scholar_lens.application.ports.FigureCheck` by trying
    each download."""

    async def unservable(self, urls: list[str]) -> list[str]:
        return await unservable(urls)
