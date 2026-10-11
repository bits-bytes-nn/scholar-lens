import asyncio
import hashlib
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import boto3
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .constants import EmbeddingModelId, LanguageModelId, LocalPaths
from .logger import logger
from .prompts import CodeAnalysisPrompt, CodebaseSummaryPrompt
from .url_guard import UnsafeUrlError, assert_url_is_public
from .utils import (
    BatchProcessor,
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
    RetryableBase,
)

DEFAULT_CHUNK_OVERLAP: int = 256
DEFAULT_CHUNK_SIZE: int = 1024
DEFAULT_MAX_CONCURRENCY: int = 10
DEFAULT_BATCH_SIZE: int = 10
EMBEDDING_BATCH_SIZE: int = 100
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
DEFAULT_MAX_CHUNKS: int = 400
# Progress is logged every this many batches while analysing/embedding.
_PROGRESS_EVERY_BATCHES: int = 5
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
        code_analysis_model_id: LanguageModelId,
        code_summarization_model_id: LanguageModelId,
        embed_model_id: EmbeddingModelId,
        paper_dir: Path | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_files: int = DEFAULT_MAX_FILES,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        region_name: str | None = None,
        profile_name: str | None = None,
        boto_session: boto3.Session | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> None:
        self.boto_session = boto_session or boto3.Session(
            region_name=region_name, profile_name=profile_name
        )
        # Route the run's TokenUsageTracker into the code-analysis and codebase-
        # summary calls so they count toward the token budget and cost metric.
        self._callbacks = list(callbacks) if callbacks else None
        self.llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self.embedding_factory = BedrockEmbeddingModelFactory(
            boto_session=self.boto_session
        )
        self._initialize_directories(paper_dir or Path.cwd())
        self.repo_paths: list[Path] | None = None
        self.vector_store: VectorStore | None = None
        self.max_files = max_files
        self.max_chunks = max_chunks

        code_analysis_llm = self.llm_factory.get_model(
            code_analysis_model_id, temperature=0.0, callbacks=self._callbacks
        )
        code_summarization_llm = self.llm_factory.get_model(
            code_summarization_model_id, temperature=0.0, callbacks=self._callbacks
        )
        self._initialize_chains(code_analysis_llm, code_summarization_llm)
        self._initialize_embeddings(embed_model_id)
        self._initialize_text_splitter(chunk_size, chunk_overlap)
        self.batch_processor = BatchProcessor(
            max_concurrency=max_concurrency, batch_size=batch_size
        )
        self.max_sequence_length = self.embedding_factory.get_max_sequence_length(
            embed_model_id
        )

    def _initialize_chains(
        self,
        code_analysis_llm: BaseLanguageModel,
        code_summarization_llm: BaseLanguageModel,
    ) -> None:
        self.code_analyser = (
            CodeAnalysisPrompt.get_prompt() | code_analysis_llm | StrOutputParser()
        )
        self.codebase_summarizer = (
            CodebaseSummaryPrompt.get_prompt()
            | code_summarization_llm
            | StrOutputParser()
        )

    def _initialize_embeddings(self, embed_model_id: EmbeddingModelId) -> None:
        self.embeddings = self.embedding_factory.get_model(embed_model_id)

    def _initialize_directories(self, paper_dir: Path) -> None:
        self.repo_cache_dir = paper_dir / LocalPaths.REPOS_DIR.value
        self._index_root = paper_dir / LocalPaths.FAISS_INDEX_DIR.value
        self.index_cache_dir = self._index_root
        for directory in [self.repo_cache_dir, self.index_cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _initialize_text_splitter(self, chunk_size: int, chunk_overlap: int) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

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
        # Key the FAISS cache by the repository set so a rerun with different
        # repositories never loads an index built from other code.
        repo_set = "\n".join(sorted(repo_urls))
        self.index_cache_dir = (
            self._index_root / hashlib.sha1(repo_set.encode("utf-8")).hexdigest()[:12]
        )
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
        from untrusted Slack input, so allow only ``https://`` to a public host
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

    async def create_or_load_index(
        self, repo_paths: list[Path] | None = None, augment: bool = True
    ) -> None:
        if self._try_load_index_sync():
            return
        repo_paths = repo_paths or self.repo_paths
        if not repo_paths:
            raise ValueError("No repository paths provided")
        documents = await self._create_documents(repo_paths)
        if not documents:
            raise NoPythonFilesError("No Python files found in repositories")
        logger.info("Created %d documents", len(documents))
        if augment:
            documents = await self._augment_documents(documents)
        await self._create_and_save_index(documents)

    def _try_load_index_sync(self) -> bool:
        if self.vector_store:
            logger.warning("Vector store already exists, re-creating.")
            self.delete_index_sync()
        try:
            self.vector_store = FAISS.load_local(
                str(self.index_cache_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Loaded index from '%s'", self.index_cache_dir)
            return True
        except Exception as e:
            logger.warning(
                "Failed to load index from '%s': %s. Creating new one.",
                self.index_cache_dir,
                str(e),
            )
            return False

    async def _create_documents(self, repo_paths: list[Path]) -> list[Document]:
        """Chunks of the most relevant files, in relevance order, capped at
        ``max_chunks`` so analysis and embedding cost stay bounded."""
        tasks = [
            self._read_file_as_document(py_file, repo_path)
            for repo_path in repo_paths
            for py_file in python_files(repo_path, max_files=self.max_files)
        ]
        doc_results = await asyncio.gather(*tasks)
        documents = [doc for doc in doc_results if doc is not None]
        chunks = self.text_splitter.split_documents(documents)
        if len(chunks) > self.max_chunks:
            logger.info(
                "Keeping the first %d of %d code chunks.", self.max_chunks, len(chunks)
            )
            chunks = chunks[: self.max_chunks]
        return chunks

    @staticmethod
    async def _read_file_as_document(
        file_path: Path, base_path: Path
    ) -> Document | None:
        try:
            # errors="replace" so a file that is valid Python but stored in
            # latin-1/cp1252 (or has a stray non-UTF-8 byte in a comment/string)
            # degrades to mostly-intact text rather than raising UnicodeDecodeError
            # and being dropped wholesale from the index + codebase summary.
            content = await asyncio.to_thread(
                file_path.read_text, encoding="utf-8", errors="replace"
            )
            return Document(
                page_content=content,
                metadata={"source": os.path.relpath(file_path, base_path)},
            )
        except Exception as e:
            logger.warning("Failed to process file '%s': %s", file_path, e)
            return None

    @staticmethod
    def _prepare_augmentation_inputs(documents: list[Document]) -> list[dict[str, str]]:
        return [{"code": doc.page_content} for doc in documents]

    async def _augment_documents(self, documents: list[Document]) -> list[Document]:
        results = await self.batch_processor.aexecute_with_fallback(
            items_to_process=documents,
            prepare_inputs_func=self._prepare_augmentation_inputs,
            batch_func=self.code_analyser.abatch,
            sequential_func=self.code_analyser.ainvoke,
            task_name="Code Document Augmentation",
        )

        augmented_docs = []
        for doc, result in zip(documents, results, strict=False):
            summary = result if result else ""
            augmented_docs.append(
                Document(
                    page_content=f"'''\n{summary}\n'''\n\n{doc.page_content}",
                    metadata=doc.metadata,
                )
            )
        return augmented_docs

    async def _create_and_save_index(self, documents: list[Document]) -> None:
        if not documents:
            logger.warning("No documents provided to create index.")
            return

        logger.info(
            "Creating FAISS index for %d documents in batches of %d.",
            len(documents),
            EMBEDDING_BATCH_SIZE,
        )

        first_batch = documents[:EMBEDDING_BATCH_SIZE]
        self.vector_store = await self._afrom_documents_with_retry(first_batch)
        logger.info("Initial vector store created with the first batch.")

        remaining_docs = documents[EMBEDDING_BATCH_SIZE:]

        total_batches = -(-len(remaining_docs) // EMBEDDING_BATCH_SIZE)
        for number, i in enumerate(
            range(0, len(remaining_docs), EMBEDDING_BATCH_SIZE), start=1
        ):
            batch = remaining_docs[i : i + EMBEDDING_BATCH_SIZE]
            if not batch:
                continue
            await self._aadd_documents_with_retry(batch)
            if number % _PROGRESS_EVERY_BATCHES == 0 or number == total_batches:
                logger.info("Embedded %d/%d document batches.", number, total_batches)

        if self.vector_store and isinstance(self.vector_store, FAISS):
            await asyncio.to_thread(
                self.vector_store.save_local, str(self.index_cache_dir)
            )
        logger.info(
            "Completed adding all batches. Index saved to '%s'",
            self.index_cache_dir,
        )

    @RetryableBase._retry("documents_embedding")
    async def _afrom_documents_with_retry(self, documents: list[Document]) -> FAISS:
        return await FAISS.afrom_documents(documents, self.embeddings)

    @RetryableBase._retry("documents_embedding")
    async def _aadd_documents_with_retry(self, documents: list[Document]) -> None:
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
        await self.vector_store.aadd_documents(documents)

    def delete_index_sync(self) -> None:
        if self.index_cache_dir.exists():
            shutil.rmtree(self.index_cache_dir)
            logger.info("Deleted index from '%s'", self.index_cache_dir)
        self.vector_store = None

    async def delete_index(self) -> None:
        if self.index_cache_dir.exists():
            await asyncio.to_thread(shutil.rmtree, self.index_cache_dir)
            logger.info("Deleted index from '%s'", self.index_cache_dir)
        self.vector_store = None

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

    async def search_similar_code(
        self, content: str, k: int = 10
    ) -> list[dict[str, float | str]]:
        if not self.vector_store:
            raise ValueError("Vector store index not created.")
        query = (
            content[: self.max_sequence_length] if self.max_sequence_length else content
        )
        results = await self.vector_store.asimilarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "similarity": float(score),
            }
            for doc, score in results
        ]


def _is_vendored_library(directory: Path, root: Path) -> bool:
    """A nested directory that is itself a packaged copy of another project
    (its own setup.py/pyproject.toml), or a known library vendored inline."""
    if directory == root:
        return False
    if any((directory / name).is_file() for name in _PROJECT_FILES):
        return True
    return directory.name.lower() in _KNOWN_LIBRARIES


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
    vendored: dict[Path, bool] = {}

    def in_vendored_dir(path: Path) -> bool:
        for parent in path.relative_to(repo_path).parents:
            directory = repo_path / parent
            if directory not in vendored:
                vendored[directory] = _is_vendored_library(directory, repo_path)
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
