"""Mirror one published post (+ its images) onto a blog branch and PR.

The GitHub side of :class:`~.publisher.Publisher`: Git Data API commits on a
deterministic branch, reuse of an already-published post's path, and cleanup of
files an earlier run of the same artifact left on the branch.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from github import GithubException, InputGitAuthor
from github import InputGitTreeElement as TreeElement

from .constants import LocalPaths
from .front_matter import split_front_matter
from .logger import logger
from .publish_types import (
    BLOG_IMAGES_DIR,
    PublishError,
    PublishRequest,
    post_key,
    post_slug,
    same_post,
)

if TYPE_CHECKING:
    from github.Repository import Repository

    from ..configs import Github as GithubConfig

_BLOB_MODE = "100644"
# An image a post shows: any reference into the blog's image folder.
_IMAGE_REF = re.compile(rf"/{re.escape(BLOG_IMAGES_DIR)}/([^\s)\"'<>?#]+)")
_MAX_HAND_EDIT_NOTES = 10


def _git_blob_sha(data: bytes) -> str:
    """The SHA git (and GitHub) assigns to a blob with ``data``."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _front_matter_date(markdown: str) -> str | None:
    """The post's front-matter ``date`` (the paper's publication time)."""
    front, _, _ = split_front_matter(markdown)
    date = (front or {}).get("date")
    return None if date is None else str(date)


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
        deletions = [
            path
            for path in self._stale_paths(markdown_path, existing, on_base, files)
            if self._only_generated(path, parent_sha)
        ]

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
                self._create_ref(branch, commit.sha)
            else:
                # Fast-forward, except when restarting the branch from base.
                try:
                    head_ref.edit(commit.sha, force=restart_from_base)
                except GithubException as e:
                    if e.status != 422:
                        raise
                    if restart_from_base:
                        raise PublishError(
                            f"GitHub refused to reset branch '{branch}' to the new "
                            f"commit ({e.status}; is the branch protected?)."
                        ) from e
                    raise PublishError(
                        f"Branch '{branch}' moved while this run updated it (another "
                        "push to the branch); rerun to rebuild on top of it."
                    ) from e
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

    def _create_ref(self, branch: str, sha: str) -> None:
        try:
            self.repo.create_git_ref(f"refs/heads/{branch}", sha)
        except GithubException as e:
            if e.status != 422:
                raise
            # "Reference already exists": a retried POST whose first attempt
            # succeeded. Move it (fast-forward only; anything else stays an error).
            try:
                self.repo.get_git_ref(f"heads/{branch}").edit(sha, force=False)
            except GithubException as moved:
                if moved.status != 422:
                    raise  # the lookup or the edit failed for another reason
                raise PublishError(
                    f"Branch '{branch}' already exists at a commit this one does not "
                    "fast-forward (another writer moved it); delete the branch or "
                    "rerun."
                ) from moved

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

        Returns the files and PR notes naming the post that is replaced and
        the hand edits to it (or its images) since it was generated.
        """
        posts = LocalPaths.POSTS_DIR.value
        new_path = f"{posts}/{markdown_path.name}"
        slug = post_slug(markdown_path.stem)
        published = sorted(
            path
            for path in on_base
            if path.startswith(f"{posts}/")
            and path.count("/") == 1
            and path.endswith(".md")
            and same_post(post_slug(Path(path).stem), slug)
        )
        if not published:
            return files, []
        old = published[0]
        notes = [
            f"Replaces the published post `{old}`.",
            *self._hand_edits(old, on_base),
        ]
        if new_path in published:
            return files, notes
        renamed = dict(files)
        renamed[old] = renamed.pop(new_path)
        logger.info("Updating the published post '%s' in place.", old)
        return renamed, notes

    def _is_generator_commit(self, commit: Any) -> bool:
        """Whether this tool made ``commit``: its author and committer are the
        configured identity. Without an email the identity is the token's
        account, which a person shares, so no commit counts as generated."""
        if not self.config.author_email:
            return False
        identity = (self.config.author_name, str(self.config.author_email))
        git = commit.commit
        return all(
            (person.name, person.email) == identity
            for person in (git.author, git.committer)
        )

    def _generated_on_base(self, commit: Any) -> bool:
        """A commit on the base branch came from this tool when it has the
        tool's identity and reached the base through a pull request from one
        of the tool's branches (a hand-fix PR uses another branch, even when a
        person commits under the same identity)."""
        prefix = f"{self.config.branch_prefix}/"
        return self._is_generator_commit(commit) and any(
            pull.head.ref.startswith(prefix) for pull in commit.get_pulls()
        )

    def _hand_edits(self, post_path: str, on_base: dict[str, str]) -> list[str]:
        """PR notes for the commits on the base branch that touched the
        published post, or an image it shows, after this tool last wrote it
        (every commit when it never did): a rerun replaces the post, so each
        hand fix must be carried over or re-applied."""
        text = self._blob_text(on_base[post_path])
        _, body, _ = split_front_matter(text)
        images = sorted(
            {
                f"{BLOG_IMAGES_DIR}/{name}"
                for name in _IMAGE_REF.findall(body)
                if f"{BLOG_IMAGES_DIR}/{name}" in on_base
            }
        )
        found: dict[str, tuple[Any, list[str]]] = {}
        try:
            for path in [post_path, *images]:
                for commit in self.repo.get_commits(
                    sha=self.config.base_branch, path=path
                ):
                    if self._generated_on_base(commit):
                        break  # older commits predate the generated version
                    found.setdefault(commit.sha, (commit, []))[1].append(path)
        except GithubException as e:
            return [
                f"Could not list the commits to `{post_path}` ({e.status}); check "
                "its history for hand edits this rerun drops."
            ]
        if not found:
            return [f"No hand edits to `{post_path}` since it was generated."]
        ordered = sorted(
            found.values(), key=lambda item: item[0].commit.committer.date, reverse=True
        )
        notes = []
        for commit, paths in ordered[:_MAX_HAND_EDIT_NOTES]:
            git = commit.commit
            headline = git.message.partition("\n")[0]
            notes.append(
                "Hand edit since generation, re-apply it if this post drops it: "
                f'{commit.sha[:7]} "{headline}" '
                f"({git.committer.date:%Y-%m-%d}) on {', '.join(paths)}: "
                f"{commit.html_url}"
            )
        if len(ordered) > _MAX_HAND_EDIT_NOTES:
            notes.append(
                f"{len(ordered) - _MAX_HAND_EDIT_NOTES} older hand edit(s) to "
                f"`{post_path}` not listed; see its history."
            )
        return notes

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
        slug = post_slug(markdown_path.stem)
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
                same_post(post_slug(Path(path).stem), slug)
                or (
                    date is not None
                    and post_key(post_slug(Path(path).stem))[0] == post_key(slug)[0]
                    and _front_matter_date(self._blob_text(branch_files[path])) == date
                )
            )
        ]
        slugs = {slug} | {post_slug(Path(path).stem) for path in old_posts}
        own_image = re.compile(
            rf"^{re.escape(BLOG_IMAGES_DIR)}/"
            rf"(?:{'|'.join(re.escape(slug) for slug in sorted(slugs))})-\d+\.(png|jpg|gif)$"
        )
        return sorted(
            [*old_posts, *(path for path in branch_only if own_image.match(path))]
        )

    def _only_generated(self, path: str, head_sha: str) -> bool:
        """Whether every commit that touched ``path`` up to ``head_sha`` is
        this tool's: a file a person added or edited on the branch is never
        removed as stale."""
        commits = list(self.repo.get_commits(sha=head_sha, path=path))
        if commits and all(self._is_generator_commit(c) for c in commits):
            return True
        logger.info("Kept '%s': not (only) written by this tool.", path)
        return False

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
        try:
            pull = self.repo.create_pull(
                title=request.pr_title,
                body=body,
                head=branch,
                base=self.config.base_branch,
            )
        except GithubException as e:
            # "A pull request already exists": a retried POST whose first
            # attempt opened it. Use that one instead of reporting a failure.
            pull = self._open_pull(branch) if e.status == 422 else None
            if pull is None:
                raise
            pull.edit(title=request.pr_title, body=body)
        logger.info("Created a pull request: %s", pull.html_url)
        return pull.html_url
