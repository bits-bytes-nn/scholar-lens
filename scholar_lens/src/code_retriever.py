import asyncio
import os
import shutil
from pathlib import Path

import boto3
import git
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStore
from tqdm import tqdm

from .constants import EmbeddingModelId, LanguageModelId, LocalPaths
from .logger import logger
from .prompts import CodeAnalysisPrompt, CodebaseSummaryPrompt
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
        region_name: str | None = None,
        profile_name: str | None = None,
        boto_session: boto3.Session | None = None,
    ) -> None:
        self.boto_session = boto_session or boto3.Session(
            region_name=region_name, profile_name=profile_name
        )
        self.llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self.embedding_factory = BedrockEmbeddingModelFactory(
            boto_session=self.boto_session
        )
        self._initialize_directories(paper_dir or Path.cwd())
        self.repo_paths: list[Path] | None = None
        self.vector_store: VectorStore | None = None

        code_analysis_llm = self.llm_factory.get_model(
            code_analysis_model_id, temperature=0.0
        )
        code_summarization_llm = self.llm_factory.get_model(
            code_summarization_model_id, temperature=0.0
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
        self.index_cache_dir = paper_dir / LocalPaths.FAISS_INDEX_DIR.value
        for directory in [self.repo_cache_dir, self.index_cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _initialize_text_splitter(self, chunk_size: int, chunk_overlap: int) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    async def download_repositories(self, repo_urls: list[str]) -> list[Path]:
        self.repo_paths = [
            self.repo_cache_dir / url.split("/")[-1].replace(".git", "")
            for url in repo_urls
        ]
        tasks = [
            self._clone_repo(url, path) for url, path in zip(repo_urls, self.repo_paths)
        ]
        await asyncio.gather(*tasks)
        return self.repo_paths

    @staticmethod
    async def _clone_repo(url: str, path: Path) -> None:
        try:
            if path.exists():
                await asyncio.to_thread(shutil.rmtree, path)
            await asyncio.to_thread(git.Repo.clone_from, url, str(path), depth=1)
            logger.info("Downloaded '%s' to '%s'", url, path)
        except Exception as e:
            logger.error("Failed to download repository '%s': %s", url, str(e))
            raise

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
        tasks = [
            self._read_file_as_document(py_file, repo_path)
            for repo_path in repo_paths
            for py_file in repo_path.rglob("*.py")
        ]
        doc_results = await asyncio.gather(*tasks)
        documents = [doc for doc in doc_results if doc is not None]
        return self.text_splitter.split_documents(documents)

    @staticmethod
    async def _read_file_as_document(
        file_path: Path, base_path: Path
    ) -> Document | None:
        try:
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
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
        for doc, result in zip(documents, results):
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

        for i in tqdm(
            range(0, len(remaining_docs), EMBEDDING_BATCH_SIZE),
            desc="Adding document batches to FAISS",
        ):
            batch = remaining_docs[i : i + EMBEDDING_BATCH_SIZE]
            if not batch:
                continue

            await self._aadd_documents_with_retry(batch)

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
        for repo_path in repo_paths:
            for py_file in repo_path.rglob("*.py"):
                try:
                    content = await asyncio.to_thread(
                        py_file.read_text, encoding="utf-8"
                    )
                    if total_chars + len(content) > self.MAX_CHARS:
                        logger.warning(
                            "Reached character limit (%d). Skipping remaining files.",
                            self.MAX_CHARS,
                        )
                        break
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

    def search_similar_code_sync(
        self, content: str, k: int = 10
    ) -> list[dict[str, float | str]]:
        if not self.vector_store:
            raise ValueError("Vector store index not created.")
        query = (
            content[: self.max_sequence_length] if self.max_sequence_length else content
        )
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "source": doc.metadata["source"],
                "content": doc.page_content,
                "similarity": float(score),
            }
            for doc, score in results
        ]

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
