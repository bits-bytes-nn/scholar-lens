"""Artifact-agnostic publishing: save markdown, upload to S3, open a blog PR.

Reviews, summaries and technical guides all produce a Markdown document plus an
optional directory of figures. This module centralises the "ship it" steps so
every artifact type publishes identically:

* write the Markdown to a local file (rewriting local image paths to the blog's
  assets layout),
* upload the post + figures to S3, and
* open a pull request against the configured github.io blog repository.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from git import Repo
from github import Auth, Github, GithubException

from .aws_helpers import S3Handler
from .constants import EnvVars, LocalPaths, S3Paths
from .logger import logger

if TYPE_CHECKING:
    from ..configs import Github as GithubConfig

_IMAGE_EXTENSIONS = [".gif", ".jpg", ".jpeg", ".png", ".svg", ".webp"]
_MARKDOWN_IMAGE_PATTERN = r"!\[(.*?)\]\((.*?)\)"


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
    rewrite_local_images: bool = True
    figures_dirname: str = LocalPaths.FIGURES_DIR.value
    extra_metadata: dict[str, str] = field(default_factory=dict)


def slugify(title: str, max_length: int = 120) -> str:
    """Filesystem- and S3-safe slug for a document title."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:max_length].strip("-")
    return slug or "untitled"


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
    ) -> None:
        self.github_config = github_config
        self.root_dir = root_dir
        self.s3_handler = s3_handler
        self.s3_bucket_name = s3_bucket_name
        self.s3_prefix = s3_prefix

    async def publish(self, request: PublishRequest) -> tuple[str | None, Path]:
        """Save + upload the artifact; return (s3_url, local_markdown_path)."""
        file_name = f"{datetime.now().strftime('%Y-%m-%d')}-{slugify(request.title)}"
        markdown = request.markdown
        if request.rewrite_local_images:
            markdown = self._rewrite_local_images(markdown, file_name)

        markdown_path = request.work_dir / f"{file_name}.md"
        await asyncio.to_thread(markdown_path.write_text, markdown, encoding="utf-8")

        s3_url = await self._upload_to_s3(request, file_name, markdown_path)
        return s3_url, markdown_path

    def _rewrite_local_images(self, markdown: str, file_name: str) -> str:
        assets_path = f"/{S3Paths.ASSETS.value}/{file_name}"

        def repl(match: re.Match) -> str:
            alt_text, img = match.group(1), match.group(2)
            if img.startswith(("http://", "https://", "/")):
                return match.group(0)
            return f"![{alt_text}]({assets_path}/{Path(img).name})"

        return re.sub(_MARKDOWN_IMAGE_PATTERN, repl, markdown)

    async def _upload_to_s3(
        self, request: PublishRequest, file_name: str, markdown_path: Path
    ) -> str | None:
        if not self.s3_handler:
            return None
        prefix = self.s3_prefix or ""
        posts_key = f"{prefix}/{S3Paths.POSTS.value}".lstrip("/")
        assets_key = f"{prefix}/{S3Paths.ASSETS.value}/{file_name}".lstrip("/")
        await self.s3_handler.upload_file_async(markdown_path, posts_key)

        figures_dir = request.work_dir / request.figures_dirname
        if figures_dir.exists():
            await asyncio.to_thread(
                self.s3_handler.upload_directory,
                figures_dir,
                assets_key,
                file_extensions=_IMAGE_EXTENSIONS,
                public_readable=self.github_config.public_assets,
            )
        return f"s3://{self.s3_bucket_name}/{posts_key}/{file_name}.md"

    async def create_pull_request(
        self, request: PublishRequest, markdown_path: Path
    ) -> None:
        repo_config = self.github_config
        if not repo_config.repo_name:
            logger.error("GitHub repository not configured.")
            return
        token = os.getenv(EnvVars.GITHUB_TOKEN.value)
        if not token:
            logger.error(
                "GitHub token not found in environment variable '%s'.",
                EnvVars.GITHUB_TOKEN.value,
            )
            return

        clone_dir = self.root_dir / LocalPaths.GITHUB_CLONE_DIR.value
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = f"{repo_config.branch_prefix}/{request.branch_id}-{timestamp}"
            await asyncio.to_thread(
                self._git_operations,
                clone_dir,
                branch_name,
                request.commit_message,
                markdown_path,
            )
            logger.info("Creating a pull request on GitHub...")
            gh_repo = Github(auth=Auth.Token(token)).get_repo(repo_config.repo_name)
            try:
                gh_repo.create_pull(
                    title=request.pr_title,
                    body=request.pr_body,
                    head=branch_name,
                    base=repo_config.base_branch,
                )
                logger.info(
                    "Successfully created a pull request: '%s'", request.pr_title
                )
            except GithubException as e:
                if e.status == 422 and "A pull request already exists" in str(e.data):
                    logger.warning(
                        "Pull request for branch '%s' already exists.", branch_name
                    )
                else:
                    raise
        except Exception as e:
            logger.error("Failed to create GitHub pull request: %s", e, exc_info=True)
        finally:
            if clone_dir.exists():
                await asyncio.to_thread(shutil.rmtree, clone_dir, ignore_errors=True)
                logger.info("Cleaned up local clone directory: '%s'", clone_dir)

    def _git_operations(
        self,
        clone_dir: Path,
        branch_name: str,
        commit_message: str,
        markdown_path: Path,
    ) -> None:
        repo_config = self.github_config
        token = os.getenv(EnvVars.GITHUB_TOKEN.value)
        repo_url = f"https://oauth2:{token}@github.com/{repo_config.repo_name}.git"
        if clone_dir.exists():
            shutil.rmtree(clone_dir)

        logger.info("Cloning repository '%s' to '%s'", repo_config.repo_name, clone_dir)
        repo = Repo.clone_from(repo_url, clone_dir)

        if branch_name in repo.heads:
            new_branch = repo.heads[branch_name]
        else:
            new_branch = repo.create_head(
                branch_name, repo.remotes.origin.refs[repo_config.base_branch]
            )
        new_branch.checkout()

        posts_dir = clone_dir / LocalPaths.POSTS_DIR.value
        posts_dir.mkdir(exist_ok=True)
        shutil.copy(markdown_path, posts_dir)
        logger.info("Copied markdown file to '%s'", posts_dir)

        figures_dir = markdown_path.parent / LocalPaths.FIGURES_DIR.value
        if figures_dir.exists():
            assets_target_dir = (
                clone_dir / LocalPaths.ASSETS_DIR.value / markdown_path.stem
            )
            assets_target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(figures_dir, assets_target_dir, dirs_exist_ok=True)
            logger.info("Copied figures to '%s'", assets_target_dir)

        if not repo.is_dirty(untracked_files=True):
            logger.warning("No changes to commit. Skipping push and pull request.")
            return

        logger.info("Committing changes...")
        repo.git.add(all=True)
        author_actor = (
            f"{repo_config.author_name} <{repo_config.author_email}>"
            if repo_config.author_email
            else repo_config.author_name
        )
        repo.git.commit("-m", commit_message, f"--author={author_actor}")

        logger.info("Pushing changes to branch '%s'...", branch_name)
        repo.remote(name="origin").push(
            refspec=f"{branch_name}:{branch_name}", force=True
        )
