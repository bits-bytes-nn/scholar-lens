import asyncio
import hashlib
import logging
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import boto3
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser

from scholar_lens.adapters.bedrock.chat_models import (
    BedrockLanguageModelFactory,
)
from scholar_lens.adapters.bedrock.prompts.extraction import (
    CodebaseSummaryPrompt,
)
from scholar_lens.adapters.http.url_guard import UnsafeUrlError, assert_url_is_public
from scholar_lens.adapters.retry import RetryableBase
from scholar_lens.domain.constants import LanguageModelId, LocalPaths
from scholar_lens.domain.usage_context import staged

logger = logging.getLogger(__name__)

# Directories never indexed: VCS metadata, virtualenvs and vendored packages
# would drown the paper's own code.
_SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "site-packages",
        "node_modules",
        "__pycache__",
        ".tox",
        "build",
        "dist",
        "third_party",
        "third-party",
        "vendor",
        "vendored",
        "external",
        "test",
        "tests",
        "testing",
        "docs",
    }
)
# Libraries that papers' repos commonly vendor whole (e.g. a patched copy of
# transformers under examples/): a directory named after one is skipped unless
# it is the repository's own package.
_KNOWN_LIBRARIES = frozenset(
    {
        "transformers",
        "torch",
        "tensorflow",
        "jax",
        "flax",
        "numpy",
        "diffusers",
        "accelerate",
        "peft",
        "datasets",
        "fairseq",
        "megatron",
        "apex",
        "deepspeed",
    }
)
_PROJECT_FILES = ("setup.py", "pyproject.toml", "setup.cfg")
MAX_FILE_BYTES: int = 100_000
DEFAULT_MAX_FILES: int = 150
_CLONE_TIMEOUT_SECONDS: int = 300
# Abort a clone that stays below 1 KB/s for 60 s, and never prompt for input.
_GIT_CLONE_ENV = {
    "GIT_HTTP_LOW_SPEED_LIMIT": "1000",
    "GIT_HTTP_LOW_SPEED_TIME": "60",
    "GIT_TERMINAL_PROMPT": "0",
}


class NoPythonFilesError(Exception):
    pass


class CodeRetriever(RetryableBase):
    MAX_CHARS: int = 200000

    def __init__(
        self,
        code_summarization_model_id: LanguageModelId,
        paper_dir: Path | None = None,
        max_files: int = DEFAULT_MAX_FILES,
        region_name: str | None = None,
        profile_name: str | None = None,
        boto_session: boto3.Session | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        llm_factory: BedrockLanguageModelFactory | None = None,
    ) -> None:
        self.boto_session = boto_session or boto3.Session(
            region_name=region_name, profile_name=profile_name
        )
        # Route the run's TokenUsageTracker into the codebase-summary call so
        # it counts toward the token budget and cost metric.
        self._callbacks = list(callbacks) if callbacks else None
        self.llm_factory = llm_factory or BedrockLanguageModelFactory(
            boto_session=self.boto_session
        )
        self._initialize_directories(paper_dir or Path.cwd())
        self.repo_paths: list[Path] | None = None
        self.max_files = max_files
        code_summarization_llm = self.llm_factory.get_model(
            code_summarization_model_id, temperature=0.0, callbacks=self._callbacks
        )
        self.codebase_summarizer = (
            CodebaseSummaryPrompt.get_prompt()
            | code_summarization_llm
            | StrOutputParser()
        )

    def _initialize_directories(self, paper_dir: Path) -> None:
        self.repo_cache_dir = paper_dir / LocalPaths.REPOS_DIR.value
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _repo_dir_name(url: str) -> str:
        """Derive a safe, collision-free cache subdirectory name from a repo URL.

        Strips trailing slashes/``.git`` and any path noise so a URL like
        ``https://github.com/org/repo/`` does not yield an empty component (which
        would make the cache path collapse onto ``repo_cache_dir`` itself — and a
        later ``rmtree`` would wipe the whole cache).

        A short hash of the *full* URL is appended so two repos with the same
        basename (``orgA/model`` and ``orgB/model``) map to distinct dirs.
        ``download_repositories`` clones concurrently and each clone does
        ``rmtree`` + ``clone_from`` on its dir; without the hash both would race
        on one path, corrupting the checkout or raising GitCommandError.
        """
        name = url.rstrip("/").split("/")[-1].removesuffix(".git").strip() or "repo"
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        return f"{name}-{digest}"

    async def download_repositories(self, repo_urls: list[str]) -> list[Path]:
        candidate_paths = [
            self.repo_cache_dir / self._repo_dir_name(url) for url in repo_urls
        ]
        results = await asyncio.gather(
            *(
                self._clone_repo(url, path)
                for url, path in zip(repo_urls, candidate_paths, strict=False)
            ),
            return_exceptions=True,
        )
        # Tolerate per-repo failures (auth, network, oversized): keep what cloned,
        # only abort if NOTHING could be downloaded.
        self.repo_paths = [
            path
            for path, result in zip(candidate_paths, results, strict=False)
            if not isinstance(result, Exception) and path.exists()
        ]
        failed = [
            url
            for url, result in zip(repo_urls, results, strict=False)
            if isinstance(result, Exception)
        ]
        if failed:
            logger.warning(
                "Failed to clone %d/%d repositories: %s",
                len(failed),
                len(repo_urls),
                ", ".join(failed),
            )
        if not self.repo_paths:
            raise NoPythonFilesError(
                "No repositories could be cloned from the supplied URLs."
            )
        return self.repo_paths

    @staticmethod
    def _assert_clonable(url: str) -> None:
        """Reject unsafe repo URLs before handing them to ``git clone``.

        ``git`` will act on ``file://`` (local exfiltration into the indexed
        corpus), ``ssh://``, and internal ``http://`` targets. Repo URLs come
        from the job request or CLI (untrusted input), so allow only ``https://`` to a public host
        (SSRF guard) — no scheme is trusted by default.
        """
        if not url.lower().startswith("https://"):
            raise UnsafeUrlError(
                f"Repository URL '{url}' must use https:// (got a non-https scheme)."
            )
        assert_url_is_public(url)

    @staticmethod
    async def _clone_repo(url: str, path: Path) -> None:
        try:
            CodeRetriever._assert_clonable(url)
            if path.exists():
                await asyncio.to_thread(shutil.rmtree, path)
            await CodeRetriever._git_clone(url, path)
            logger.info("Downloaded '%s' to '%s'", url, path)
        except Exception as e:
            logger.error("Failed to download repository '%s': %s", url, str(e))
            raise

    @staticmethod
    async def _git_clone(url: str, path: Path) -> None:
        """Shallow, single-branch clone with no symlinks checked out (a symlink
        could point the indexer at local files) and no HTTP redirects (the URL
        was SSRF-checked, its redirect target was not). The git process is
        killed when it exceeds :data:`_CLONE_TIMEOUT_SECONDS`."""
        process = await asyncio.create_subprocess_exec(
            "git",
            "-c",
            "core.symlinks=false",
            "-c",
            "http.followRedirects=false",
            "clone",
            "--depth=1",
            "--single-branch",
            "--",
            url,
            str(path),
            env={**os.environ, **_GIT_CLONE_ENV},
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_CLONE_TIMEOUT_SECONDS
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"git clone of '{url}' exceeded {_CLONE_TIMEOUT_SECONDS}s"
            ) from None
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()[-500:]
            raise RuntimeError(f"git clone of '{url}' failed: {message}")

    @staged("code_summary")
    async def generate_codebase_summary(
        self, repo_paths: list[Path] | None = None
    ) -> str:
        repo_paths = repo_paths or self.repo_paths
        if not repo_paths:
            raise ValueError("No repository paths provided")
        all_content, total_chars = [], 0
        limit_reached = False
        for repo_path in repo_paths:
            if limit_reached:
                break
            for py_file in python_files(repo_path, max_files=self.max_files):
                try:
                    # errors="replace" (see _read_file_as_document): don't drop a
                    # non-UTF-8-clean source file from the codebase summary.
                    content = await asyncio.to_thread(
                        py_file.read_text, encoding="utf-8", errors="replace"
                    )
                    if total_chars + len(content) > self.MAX_CHARS:
                        if total_chars >= self.MAX_CHARS * 0.95:
                            logger.warning(
                                "Reached character limit (%d); skipping the rest.",
                                self.MAX_CHARS,
                            )
                            limit_reached = True
                            break
                        # One oversized file must not end the summary early.
                        logger.info(
                            "Skipping large file '%s' for the summary.", py_file
                        )
                        continue
                    total_chars += len(content)
                    all_content.append(content)
                except Exception as e:
                    logger.warning("Failed to read file '%s': %s", py_file, e)
        if not all_content:
            raise NoPythonFilesError("No Python files found in repositories")
        summary = await self._summarize_codebase("\n\n".join(all_content))
        return summary

    @RetryableBase._retry("codebase_summarization")
    async def _summarize_codebase(self, codebase: str) -> str:
        return await self.codebase_summarizer.ainvoke({"codebase": codebase})


# A project's declared name: name = "peft" (pyproject/setup.cfg) or name="peft".
_DECLARED_NAME = re.compile(r"""^\s*name\s*=\s*["']?([A-Za-z0-9_.-]+)""", re.M)
_SETUP_PY_NAME = re.compile(r"""\bname\s*=\s*["']([A-Za-z0-9_.-]+)["']""")


def _name_tokens(name: str) -> set[str]:
    normalized = re.sub(r"[-_.]+", "_", name.lower()).strip("_")
    return {normalized, *normalized.split("_")} - {""}


def own_package_names(root: Path) -> set[str]:
    """Names the repository's own package may go by: its directory name (the
    clone folder ends in "-<hash>") and the name its project file declares, so
    a repo of peft or Megatron-LM indexes its own ``peft/`` or ``megatron/``."""
    names = _name_tokens(re.sub(r"-[0-9a-f]{8}$", "", root.name))
    for project_file in _PROJECT_FILES:
        path = root / project_file
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pattern = _SETUP_PY_NAME if project_file == "setup.py" else _DECLARED_NAME
        if match := pattern.search(text):
            names |= _name_tokens(match.group(1))
    return names


def _is_vendored_library(directory: Path, root: Path, own: set[str]) -> bool:
    """A nested directory that is itself a packaged copy of another project
    (its own setup.py/pyproject.toml), or a known library vendored inline;
    never the repository's own package."""
    if directory == root or directory.parent == root / "src":
        return False  # the root, or a package in the src layout
    if any((directory / name).is_file() for name in _PROJECT_FILES):
        return True
    name = directory.name.lower()
    return name in _KNOWN_LIBRARIES and name not in own


def _readme_text(root: Path) -> str:
    for candidate in sorted(root.glob("README*")):
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                return ""
    return ""


def python_files(repo_path: Path, *, max_files: int | None = None) -> list[Path]:
    """The repository's own Python sources, most relevant first.

    Skips symlinks, files resolving outside the repository, files over
    :data:`MAX_FILE_BYTES`, VCS/virtualenv/test/doc directories and vendored
    copies of other libraries. Files the README mentions (by module or package
    name) come first, then shallower paths; ``max_files`` caps the result.
    """
    root = repo_path.resolve()
    readme = _readme_text(repo_path)
    own = own_package_names(repo_path)
    vendored: dict[Path, bool] = {}

    def in_vendored_dir(path: Path) -> bool:
        for parent in path.relative_to(repo_path).parents:
            directory = repo_path / parent
            if directory not in vendored:
                vendored[directory] = _is_vendored_library(directory, repo_path, own)
            if vendored[directory]:
                return True
        return False

    candidates = []
    for path in sorted(repo_path.rglob("*.py")):
        relative = path.relative_to(repo_path)
        if any(part.lower() in _SKIPPED_DIRS for part in relative.parts[:-1]):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        if in_vendored_dir(path):
            continue
        names = {relative.stem.lower(), *(p.lower() for p in relative.parts[:-1])}
        mentioned = any(len(n) >= 3 and n in readme for n in names - {"__init__"})
        candidates.append((not mentioned, len(relative.parts), str(relative), path))
    candidates.sort(key=lambda c: c[:3])
    files = [c[3] for c in candidates]
    if max_files is not None and len(files) > max_files:
        logger.info(
            "Indexing the %d most relevant of %d Python files in '%s'.",
            max_files,
            len(files),
            repo_path.name,
        )
        files = files[:max_files]
    return files
