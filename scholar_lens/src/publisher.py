"""Artifact-agnostic publishing: save markdown, upload to S3, open a blog PR.

Reviews, summaries and technical guides all produce a Markdown document plus an
optional directory of figures. This module centralises the "ship it" steps so
every artifact type publishes identically:

* localize images: arXiv-hosted and local figures are stored flat as
  ``assets/images/<slug>-<n>.<ext>`` and referenced by absolute path, the
  blog's convention (remote arXiv links 404 when versions change and get no
  layout dimensions),
* write the Markdown to a local file and upload the post + images to S3, and
* commit the post + images to a deterministic branch of the configured
  github.io repository through the GitHub API and open (or update) its PR.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from github import Auth, Github, GithubException, GithubRetry, InputGitAuthor
from github import InputGitTreeElement as TreeElement

from .aws_helpers import S3Handler
from .constants import EnvVars, LocalPaths, S3Paths
from .logger import logger
from .markdown_lint import lint_markdown, writing_notes
from .markdown_math import SEGMENT_PATTERN, normalize_math
from .pinned_transport import PinnedHTTPTransport
from .post_conventions import PostConventions, load_local_conventions
from .post_validator import (
    ARXIV_IMAGE_HOSTS,
    ArtifactKind,
    PostValidationError,
    validate_post,
)
from .prompt_safety import strip_defanging
from .url_guard import UnsafeUrlError, assert_url_is_public
from .utils.images import is_svg, svg_to_png

if TYPE_CHECKING:
    from github.Repository import Repository

    from ..configs import Github as GithubConfig

# Blog image convention: one flat folder, referenced by absolute path.
BLOG_IMAGES_DIR = "assets/images"
# Scratch folder (inside the artifact's work_dir) holding the localized images
# that are uploaded to S3 and committed to the blog, one sub-folder per post
# slug: a paper's summary and review share the work_dir.
_STAGED_IMAGES_DIRNAME = "blog-images"
SUMMARY_SLUG_SUFFIX = "-summary"
# Hosts whose images must be copied into the blog (remote arXiv/ar5iv links
# break when paper versions change and get no layout dimensions).
_ARXIV_HOSTS = ARXIV_IMAGE_HOSTS
_IMAGE_DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
# Accepted image formats by magic bytes -> stored extension.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
)
# Image target tolerates one level of parens in the URL/path (e.g. Wikipedia's
# ``File_(diagram).png`` or a scratch path ``fig_(a).png``). A naive ``(.*?)``
# stops at the first ``)``, truncating the URL — the tail leaks as prose and the
# rewritten local path 404s. The balanced group ``[^()]*(\([^()]*\)[^()]*)*``
# still stops at the real closing paren, so multiple images per line split fine.
_MARKDOWN_IMAGE_PATTERN = (
    r"!\[(.*?)\]\(([^()\s]*(?:\([^()]*\)[^()\s]*)*)(\s+\"[^\"]*\")?\)"
)

# GitHub API client limits: per-request timeout (seconds) and a small retry
# budget (GithubRetry also waits out secondary rate limits).
_GITHUB_TIMEOUT_SECONDS = 30
_GITHUB_MAX_RETRIES = 3
# Post files are named "<YYYY-MM-DD>-<slug>.md" (see Publisher.publish).
_DATED_POST = re.compile(r"^(\d{4}-\d{2}-\d{2})-(?P<slug>.+)$")
_BLOB_MODE = "100644"


class PublishError(Exception):
    """Publishing to the blog repository failed."""


class PostAssetError(Exception):
    """An image the post references could not be localized; nothing is saved,
    uploaded or proposed."""


@dataclass
class PublishRequest:
    """Everything needed to publish one artifact."""

    title: str
    markdown: str
    work_dir: Path
    branch_id: str
    pr_title: str
    pr_body: str
    commit_message: str
    # Artifact kind the post is validated as ("review" | "summarize" | "guide").
    kind: ArtifactKind = "review"
    # Pipeline-side findings for the reviewer (e.g. a tag the paper never
    # mentions); shown in the PR checklist with the validator's warnings.
    review_notes: list[str] = field(default_factory=list)
    # Filled by Publisher.publish: non-blocking post-validation findings, shown
    # as a checklist in the PR body.
    validation_warnings: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        """Post slug: a summary and a review of one paper must not share a
        file name, image names or URL."""
        base = slugify(self.title)
        return f"{base}{SUMMARY_SLUG_SUFFIX}" if self.kind == "summarize" else base

    @property
    def staged_images_dir(self) -> Path:
        """This post's localized images; a rerun replaces only these."""
        return self.work_dir / _STAGED_IMAGES_DIRNAME / self.slug

    def pr_body_with_checklist(self, extra_notes: Sequence[str] = ()) -> str:
        return with_review_checklist(
            self.pr_body, [*self.validation_warnings, *extra_notes]
        )


def _split_front_matter(markdown: str) -> tuple[str, str]:
    """``(front_matter_with_delimiters, body)``; front matter may be empty."""
    if markdown.startswith("---\n"):
        end = markdown.find("\n---\n", 4)
        if end != -1:
            return markdown[: end + 5], markdown[end + 5 :]
    return "", markdown


def slugify(title: str, max_length: int = 80) -> str:
    """Filesystem- and S3-safe ASCII slug for a document title, cut at a word
    boundary (the blog URL comes from it, so keep it readable and stable).

    A title whose letters are all non-ASCII (e.g. Korean) gets a stable hash
    suffix, so two such titles never share a file, S3 key or branch.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(slug) > max_length:
        cut = slug[: max_length + 1]
        slug = (cut.rsplit("-", 1)[0] if "-" in cut else cut[:max_length]).strip("-")
    if slug:
        return slug
    if not any(ch.isalnum() for ch in title):
        return "untitled"
    digest = hashlib.sha1(title.strip().encode("utf-8")).hexdigest()[:8]
    return f"untitled-{digest}"


# Footer shared by every auto-generated blog PR body.
_PR_FOOTER = "This pull request was automatically generated by the Scholar-Lens system."


def build_pr_body(
    *,
    kind: str,
    title: str,
    fields: dict[str, str],
    warnings: Sequence[str] = (),
) -> str:
    """Build a consistent blog-PR description across all artifact types.

    ``kind`` is the human label ("paper review" / "paper summary" / "technical
    guide"); ``fields`` are the artifact-specific metadata rows (rendered as
    ``- **Key**: value``), in insertion order. Empty-valued fields are skipped so
    each artifact only shows what's relevant, while the header/footer stay
    identical. ``warnings`` become a review checklist."""
    header = f"This PR adds an AI-generated {kind}: **{title}**"
    rows = "\n".join(f"- **{k}**: {v}" for k, v in fields.items() if v)
    return with_review_checklist(f"{header}\n\n{rows}\n\n{_PR_FOOTER}", warnings)


def with_review_checklist(body: str, warnings: Sequence[str]) -> str:
    """Append post-validation warnings to a PR body as unchecked items."""
    if not warnings:
        return body
    items = "\n".join(f"- [ ] {w}" for w in warnings)
    return f"{body}\n\n### Review checklist\n{items}"


def is_arxiv_host(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == h or host.endswith(f".{h}") for h in _ARXIV_HOSTS)


def image_format(data: bytes) -> str | None:
    """Stored extension for a PNG/JPEG payload; None for anything else."""
    for signature, extension in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return extension
    return None


def download_image(url: str) -> bytes:
    """Fetch an image over the SSRF-pinned transport, with timeouts and a size
    cap. Every redirect hop is re-validated by the transport."""
    assert_url_is_public(url)
    with httpx.Client(
        transport=PinnedHTTPTransport(),
        timeout=_IMAGE_DOWNLOAD_TIMEOUT,
        follow_redirects=True,
        max_redirects=5,
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > _MAX_IMAGE_BYTES:
                    raise ValueError(f"image exceeds {_MAX_IMAGE_BYTES} bytes")
                chunks.append(chunk)
    return b"".join(chunks)


class ImageLocalizer:
    """Rewrite every image in a post to the blog's flat ``/assets/images``
    layout, writing the image files into ``staging_dir``.

    * arXiv-family URLs are downloaded and must be PNG/JPEG,
    * local figures (bare names or scratch paths) are read from
      ``figures_dir``,
    * paths already under ``/assets/`` are left alone,
    * other remote images stay external unless ``download_external``.

    Images are named ``<slug>-<n>.<ext>`` in order of first appearance; the
    same source reused in the post maps to the same file. Any image that
    cannot be localized raises :class:`PostAssetError`.
    """

    def __init__(
        self,
        *,
        slug: str,
        figures_dir: Path,
        staging_dir: Path,
        download_external: bool = False,
        fetch: Callable[[str], bytes] = download_image,
    ) -> None:
        self.slug = slug
        self.figures_dir = figures_dir
        self.staging_dir = staging_dir
        self.download_external = download_external
        self.fetch = fetch

    def localize(self, markdown: str) -> str:
        if self.staging_dir.exists():
            # This post's own scratch folder: drop a previous run's images.
            shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True)
        hosted: dict[str, str] = {}
        failures: list[str] = []

        code = [
            (m.start(), m.end())
            for m in SEGMENT_PATTERN.finditer(markdown)
            if m.lastgroup in ("fence", "icode")
        ]

        def repl(match: re.Match) -> str:
            if any(start <= match.start() < end for start, end in code):
                return match.group(0)  # an image example inside code
            alt_text, src = match.group(1), match.group(2).strip()
            title = match.group(3) or ""
            if src in hosted:
                return f"![{alt_text}]({hosted[src]}{title})"
            try:
                data = self._load(src)
            except PostAssetError as e:
                failures.append(str(e))
                return match.group(0)
            if data is None:
                return match.group(0)  # left as-is by policy
            if is_svg(data):
                try:
                    data = svg_to_png(data)  # the blog serves PNG/JPEG only
                except (ValueError, UnicodeDecodeError) as e:
                    failures.append(
                        f"'{src}' is an SVG that could not be rendered: {e}"
                    )
                    return match.group(0)
            extension = image_format(data)
            if extension is None:
                failures.append(f"'{src}' is not a PNG or JPEG image")
                return match.group(0)
            name = f"{self.slug}-{len(hosted) + 1}.{extension}"
            (self.staging_dir / name).write_bytes(data)
            hosted[src] = f"/{BLOG_IMAGES_DIR}/{name}"
            return f"![{alt_text}]({hosted[src]}{title})"

        markdown = re.sub(_MARKDOWN_IMAGE_PATTERN, repl, markdown)
        if failures:
            raise PostAssetError("; ".join(failures))
        return markdown

    def _load(self, src: str) -> bytes | None:
        if src.startswith(("http://", "https://")):
            if not (is_arxiv_host(src) or self.download_external):
                return None
            try:
                return self.fetch(src)
            except (httpx.HTTPError, UnsafeUrlError, ValueError) as e:
                raise PostAssetError(f"could not download '{src}': {e}") from e
        if src.startswith("/assets/") or src.startswith("data:"):
            return None
        local = self.figures_dir / Path(src).name
        if not local.is_file():
            raise PostAssetError(f"figure '{src}' not found in {self.figures_dir}")
        return local.read_bytes()


class Publisher:
    """Publishes artifacts to S3 and the configured blog repository."""

    def __init__(
        self,
        github_config: GithubConfig,
        *,
        root_dir: Path,
        s3_handler: S3Handler | None = None,
        s3_bucket_name: str | None = None,
        s3_prefix: str = "",
        conventions: PostConventions | None = None,
        download_external_images: bool = False,
        fetch_image: Callable[[str], bytes] = download_image,
    ) -> None:
        self.github_config = github_config
        self.conventions = conventions or load_local_conventions()
        self.root_dir = root_dir
        self.s3_handler = s3_handler
        self.s3_bucket_name = s3_bucket_name
        self.s3_prefix = s3_prefix
        self.download_external_images = download_external_images
        self.fetch_image = fetch_image

    async def publish(self, request: PublishRequest) -> tuple[str | None, Path]:
        """Normalize, localize images, validate, save and upload the post.

        Returns ``(s3_url, local_markdown_path)``. Raises
        :class:`PostAssetError` (or the validator's error) before anything is
        written when the post cannot be published as-is.
        """
        slug = request.slug
        # The post filename is prefixed with the publish date (today); the paper's
        # own date lives in the front-matter `date` field, set by the caller.
        file_name = f"{datetime.now(UTC).strftime('%Y-%m-%d')}-{slug}"
        # Body-only passes: the front matter is already YAML-escaped and must
        # not be math-normalized or HTML-escaped.
        # Defanged prompt tags a model copied from its input get their plain
        # "<" back everywhere (titles too; harmless inside YAML strings).
        front_matter, body = _split_front_matter(strip_defanging(request.markdown))
        # Blog math rules ($$ only, no \_ in math, blank lines around display
        # blocks), then the other kramdown/MathJax auto-fixes.
        body = lint_markdown(normalize_math(body))
        markdown = front_matter + body
        localizer = ImageLocalizer(
            slug=slug,
            figures_dir=request.work_dir / LocalPaths.FIGURES_DIR.value,
            staging_dir=request.staged_images_dir,
            download_external=self.download_external_images,
            fetch=self.fetch_image,
        )
        markdown = await asyncio.to_thread(localizer.localize, markdown)
        try:
            request.validation_warnings = [
                *request.review_notes,
                # Rule slips the generators could not fix, on the final text.
                *writing_notes(body),
                *self._validate(markdown, request),
            ]
        except PostValidationError as e:
            if e.rejected_path and self.s3_handler:
                # In Batch the work dir is container scratch; keep a copy. A
                # failed copy must not hide the validation result.
                key = f"{self.s3_prefix}/rejected".lstrip("/")
                try:
                    await self.s3_handler.upload_file_async(Path(e.rejected_path), key)
                    e.rejected_s3_uri = (
                        f"s3://{self.s3_bucket_name}/{key}/{Path(e.rejected_path).name}"
                    )
                except Exception as upload_error:  # noqa: BLE001
                    logger.warning("Could not copy the rejected post: %s", upload_error)
            raise

        markdown_path = request.work_dir / f"{file_name}.md"
        await asyncio.to_thread(markdown_path.write_text, markdown, encoding="utf-8")

        s3_url = await self._upload_to_s3(request, markdown_path)
        return s3_url, markdown_path

    def _validate(self, markdown: str, request: PublishRequest) -> list[str]:
        """Check the finished post against the blog conventions.

        Errors stop publishing (nothing saved, uploaded or proposed); the post
        is kept beside the work dir as ``*.rejected.md`` with the violations so
        it can be inspected. Warnings are returned for the PR checklist.
        """
        violations = validate_post(markdown, self.conventions, request.kind)
        if blocking := [v for v in violations if v.level == "error"]:
            rejected = request.work_dir / f"{request.slug}.rejected.md"
            report = "\n".join(
                [
                    *(f"- {v}" for v in violations),
                    *(f"- note: {n}" for n in request.review_notes),
                ]
            )
            rejected.write_text(
                f"{markdown}\n\n<!-- violations\n{report}\n-->\n", encoding="utf-8"
            )
            logger.error(
                "Post rejected with %d error(s); kept at '%s'", len(blocking), rejected
            )
            error = PostValidationError(violations)
            error.rejected_path = str(rejected)
            raise error
        return [str(v) for v in violations]

    async def _upload_to_s3(
        self, request: PublishRequest, markdown_path: Path
    ) -> str | None:
        if not self.s3_handler:
            return None
        prefix = self.s3_prefix or ""
        posts_key = f"{prefix}/{S3Paths.POSTS.value}".lstrip("/")
        images_key = f"{prefix}/{BLOG_IMAGES_DIR}".lstrip("/")
        await self.s3_handler.upload_file_async(markdown_path, posts_key)

        staged = request.staged_images_dir
        if staged.is_dir() and any(staged.iterdir()):
            await asyncio.to_thread(
                self.s3_handler.upload_directory,
                staged,
                images_key,
                public_readable=self.github_config.public_assets,
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
        repo_name = self.github_config.repo_name
        if not repo_name:
            raise PublishError("resources.github.repo_name is not configured.")
        token = os.getenv(EnvVars.GITHUB_TOKEN.value)
        if not token:
            raise PublishError(f"{EnvVars.GITHUB_TOKEN.value} is not set.")
        try:
            # PyGithub is blocking HTTP; keep it off the event loop.
            return await asyncio.to_thread(
                self._sync_pull_request, token, repo_name, request, markdown_path
            )
        except PublishError:
            raise
        except Exception as e:
            raise PublishError(str(self._redact_token(e, token))) from None

    def _sync_pull_request(
        self, token: str, repo_name: str, request: PublishRequest, markdown_path: Path
    ) -> str:
        client = Github(
            auth=Auth.Token(token),
            timeout=_GITHUB_TIMEOUT_SECONDS,
            retry=GithubRetry(total=_GITHUB_MAX_RETRIES),
        )
        try:
            return BlogBranchSync(client.get_repo(repo_name), self.github_config).sync(
                request, markdown_path
            )
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


def _git_blob_sha(data: bytes) -> str:
    """The SHA git (and GitHub) assigns to a blob with ``data``."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# The `date:` key inside the leading front matter block.
_FRONT_MATTER_DATE = re.compile(r"\A---\n(?:(?!---).*\n)*?date: *(?P<date>.+)\n")


def _front_matter_date(markdown: str) -> str | None:
    match = _FRONT_MATTER_DATE.match(markdown)
    return match.group("date").strip() if match else None


def _post_slug(stem: str) -> str:
    match = _DATED_POST.match(stem)
    return match.group("slug") if match else stem


def _same_post(slug: str, other: str) -> bool:
    """Slugs naming the same post: older posts kept the double hyphen an
    earlier slugify left for ": " ("direct-preference-optimization--your-…")."""
    return re.sub(r"-{2,}", "-", slug) == re.sub(r"-{2,}", "-", other)


def _is_summary_slug(slug: str) -> bool:
    """Summary posts carry this suffix (see :attr:`PublishRequest.slug`)."""
    return slug.endswith(SUMMARY_SLUG_SUFFIX)


class BlogBranchSync:
    """Mirror one post (+ its images) onto a dedicated branch and PR.

    The branch is ``<branch_prefix>/<branch_id>`` — deterministic, so a rerun
    (Batch retry, regenerated post) updates the same branch and PR instead of
    opening a duplicate. While the PR is open, new commits build on the branch
    head, preserving manual edits made on it; otherwise (merged, squash-merged
    or closed) the branch is restarted from the base.
    """

    def __init__(self, repo: Repository, config: GithubConfig) -> None:
        self.repo = repo
        self.config = config

    def sync(self, request: PublishRequest, markdown_path: Path) -> str:
        branch = f"{self.config.branch_prefix}/{request.branch_id}"
        base_sha = self.repo.get_git_ref(f"heads/{self.config.base_branch}").object.sha
        head_ref = self._find_ref(branch)
        open_pull = self._open_pull(branch) if head_ref is not None else None
        parent_sha = base_sha
        restart_from_base = False
        if head_ref is not None:
            head_sha = head_ref.object.sha
            if open_pull is not None and self.repo.compare(base_sha, head_sha).ahead_by:
                parent_sha = head_sha
            else:
                restart_from_base = head_sha != base_sha

        base_tree = self._tree(base_sha)
        parent_tree = base_tree if parent_sha == base_sha else self._tree(parent_sha)
        on_base = {e.path: e.sha for e in base_tree.tree if e.type == "blob"}
        existing = {e.path: e.sha for e in parent_tree.tree if e.type == "blob"}
        files = self._artifact_files(request, markdown_path)
        files, notes = self._keep_published_path(files, markdown_path, on_base)
        changes = [
            path
            for path, data in files.items()
            if existing.get(path) != _git_blob_sha(data)
        ]
        deletions = self._stale_paths(markdown_path, existing, on_base, files)

        if changes or deletions:
            elements = [
                TreeElement(
                    path,
                    _BLOB_MODE,
                    "blob",
                    sha=self.repo.create_git_blob(
                        base64.b64encode(files[path]).decode("ascii"), "base64"
                    ).sha,
                )
                for path in changes
            ]
            elements += [
                TreeElement(path, _BLOB_MODE, "blob", sha=None) for path in deletions
            ]
            tree = self.repo.create_git_tree(elements, base_tree=parent_tree)
            commit = self.repo.create_git_commit(
                request.commit_message,
                tree,
                [self.repo.get_git_commit(parent_sha)],
                **self._identity(),
            )
            if head_ref is None:
                self.repo.create_git_ref(f"refs/heads/{branch}", commit.sha)
            else:
                # Fast-forward, except when restarting the branch from base.
                head_ref.edit(commit.sha, force=restart_from_base)
            logger.info(
                "Committed %d file(s), removed %d stale file(s) on '%s'.",
                len(changes),
                len(deletions),
                branch,
            )
        elif parent_sha == base_sha:
            raise PublishError(
                f"'{self.config.base_branch}' already contains this post unchanged; "
                "no pull request needed."
            )
        else:
            logger.info("Branch '%s' already has this exact content.", branch)
        return self._open_or_update_pull(branch, request, open_pull, notes)

    def _find_ref(self, branch: str) -> Any:
        try:
            return self.repo.get_git_ref(f"heads/{branch}")
        except GithubException as e:
            if e.status == 404:
                return None
            raise

    def _tree(self, commit_sha: str) -> Any:
        tree = self.repo.get_git_tree(
            self.repo.get_git_commit(commit_sha).tree.sha, recursive=True
        )
        if tree.raw_data.get("truncated"):
            # A partial listing would hide published posts and stale files.
            raise PublishError("The blog repository tree is too large to list.")
        return tree

    def _keep_published_path(
        self, files: dict[str, bytes], markdown_path: Path, on_base: dict[str, str]
    ) -> tuple[dict[str, bytes], list[str]]:
        """A post already on the base branch keeps its file name (and so its
        date prefix and URL): a rerun updates it instead of adding a copy.

        Returns the files and a PR note naming the post that is replaced.
        """
        posts = LocalPaths.POSTS_DIR.value
        new_path = f"{posts}/{markdown_path.name}"
        slug = _post_slug(markdown_path.stem)
        published = sorted(
            path
            for path in on_base
            if path.startswith(f"{posts}/")
            and path.count("/") == 1
            and path.endswith(".md")
            and _same_post(_post_slug(Path(path).stem), slug)
        )
        if not published:
            return files, []
        note = (
            f"Replaces the published post `{published[0]}`; "
            "check that no hand edits to it are lost."
        )
        if new_path in published:
            return files, [note]
        renamed = dict(files)
        renamed[published[0]] = renamed.pop(new_path)
        logger.info("Updating the published post '%s' in place.", published[0])
        return renamed, [note]

    @staticmethod
    def _artifact_files(
        request: PublishRequest, markdown_path: Path
    ) -> dict[str, bytes]:
        """Repository path -> bytes for the post and its localized images."""
        files = {
            f"{LocalPaths.POSTS_DIR.value}/{markdown_path.name}": markdown_path.read_bytes()
        }
        staged = request.staged_images_dir
        if staged.is_dir():
            for path in sorted(staged.iterdir()):
                if path.is_file():
                    files[f"{BLOG_IMAGES_DIR}/{path.name}"] = path.read_bytes()
        return files

    def _stale_paths(
        self,
        markdown_path: Path,
        branch_files: dict[str, str],
        on_base: dict[str, str],
        new_files: dict[str, bytes],
    ) -> list[str]:
        """Files of earlier runs that exist only on this artifact's branch:
        older copies of the post and numbered images this run no longer
        produces.

        An older copy has the post's slug or, after a retitled arXiv version,
        the same front-matter ``date`` (the paper's publication time) and the
        same kind (a summary slug ends in ``-summary``, so a review and a
        summary of one paper never replace each other). Any other post on the
        branch was added by hand and is kept; a hand-added post of the same
        kind with that exact timestamp would be taken for an older copy.
        """
        posts = LocalPaths.POSTS_DIR.value
        slug = _post_slug(markdown_path.stem)
        date = _front_matter_date(markdown_path.read_text(encoding="utf-8"))
        branch_only = [
            path
            for path in branch_files
            if path not in on_base and path not in new_files
        ]
        old_posts = [
            path
            for path in branch_only
            if path.startswith(f"{posts}/")
            and path.count("/") == 1
            and path.endswith(".md")
            and (
                _post_slug(Path(path).stem) == slug
                or (
                    date is not None
                    and _is_summary_slug(_post_slug(Path(path).stem))
                    == _is_summary_slug(slug)
                    and _front_matter_date(self._blob_text(branch_files[path])) == date
                )
            )
        ]
        slugs = {slug} | {_post_slug(Path(path).stem) for path in old_posts}
        own_image = re.compile(
            rf"^{re.escape(BLOG_IMAGES_DIR)}/"
            rf"(?:{'|'.join(re.escape(slug) for slug in sorted(slugs))})-\d+\.(png|jpg)$"
        )
        return sorted(
            [*old_posts, *(path for path in branch_only if own_image.match(path))]
        )

    def _blob_text(self, sha: str) -> str:
        blob = self.repo.get_git_blob(sha)
        return base64.b64decode(blob.content).decode("utf-8", errors="replace")

    def _identity(self) -> dict[str, InputGitAuthor]:
        """Author/committer from config; without an email GitHub attributes the
        commit to the token's account."""
        if not self.config.author_email:
            return {}
        actor = InputGitAuthor(self.config.author_name, str(self.config.author_email))
        return {"author": actor, "committer": actor}

    def _open_pull(self, branch: str) -> Any:
        owner = self.repo.full_name.split("/")[0]
        for pull in self.repo.get_pulls(
            state="open", head=f"{owner}:{branch}", base=self.config.base_branch
        ):
            return pull
        return None

    def _open_or_update_pull(
        self, branch: str, request: PublishRequest, pull: Any, notes: list[str]
    ) -> str:
        body = request.pr_body_with_checklist(notes)
        if pull is not None:
            pull.edit(title=request.pr_title, body=body)
            logger.info("Updated the existing pull request: %s", pull.html_url)
            return pull.html_url
        pull = self.repo.create_pull(
            title=request.pr_title,
            body=body,
            head=branch,
            base=self.config.base_branch,
        )
        logger.info("Created a pull request: %s", pull.html_url)
        return pull.html_url
