"""The sections engine's searchable code index: the paper's repositories
chunked, described by a model and embedded into an in-memory vector store
(``langchain_core``'s, ranked by cosine similarity) the review sections
search for the code they explain."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import boto3
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scholar_lens.adapters.bedrock.chat_models import BedrockLanguageModelFactory
from scholar_lens.adapters.code.retriever import (
    DEFAULT_MAX_FILES,
    CodeRetriever,
    NoPythonFilesError,
    python_files,
)
from scholar_lens.adapters.retry import RetryableBase
from scholar_lens.domain.constants import EmbeddingModelId, LanguageModelId
from scholar_lens.domain.prompt_safety import neutralize_prompt_tags
from scholar_lens.domain.usage_context import staged
from scholar_lens.legacy.batch import BatchProcessor
from scholar_lens.legacy.embeddings import BedrockEmbeddingModelFactory
from scholar_lens.legacy.prompts.extraction import CodeAnalysisPrompt

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_OVERLAP: int = 256
DEFAULT_CHUNK_SIZE: int = 1024
DEFAULT_MAX_CONCURRENCY: int = 10
DEFAULT_BATCH_SIZE: int = 10
EMBEDDING_BATCH_SIZE: int = 100
DEFAULT_MAX_CHUNKS: int = 400
# Progress is logged every this many batches while analysing/embedding.
_PROGRESS_EVERY_BATCHES: int = 5


class CodeIndex(CodeRetriever):
    """A :class:`CodeRetriever` that also indexes the code for search."""

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
        llm_factory: BedrockLanguageModelFactory | None = None,
        embedding_factory: BedrockEmbeddingModelFactory | None = None,
    ) -> None:
        super().__init__(
            code_summarization_model_id,
            paper_dir=paper_dir,
            max_files=max_files,
            region_name=region_name,
            profile_name=profile_name,
            boto_session=boto_session,
            callbacks=callbacks,
            llm_factory=llm_factory,
        )
        self.embedding_factory = embedding_factory or BedrockEmbeddingModelFactory(
            boto_session=self.boto_session
        )
        self.vector_store: VectorStore | None = None
        self.max_chunks = max_chunks
        code_analysis_llm = self.llm_factory.get_model(
            code_analysis_model_id, temperature=0.0, callbacks=self._callbacks
        )
        self.code_analyser = (
            CodeAnalysisPrompt.get_prompt() | code_analysis_llm | StrOutputParser()
        )
        self._initialize_embeddings(embed_model_id)
        self._initialize_text_splitter(chunk_size, chunk_overlap)
        self.batch_processor = BatchProcessor(
            max_concurrency=max_concurrency, batch_size=batch_size
        )
        self.max_sequence_length = self.embedding_factory.get_max_sequence_length(
            embed_model_id
        )

    def _initialize_embeddings(self, embed_model_id: EmbeddingModelId) -> None:
        self.embeddings = self.embedding_factory.get_model(embed_model_id)

    def _initialize_text_splitter(self, chunk_size: int, chunk_overlap: int) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    async def build_index(
        self, repo_paths: list[Path] | None = None, augment: bool = True
    ) -> None:
        """Embed the repositories' code into an in-memory index for this run
        (it is never persisted: a run's repositories and settings decide it)."""
        repo_paths = repo_paths or self.repo_paths
        if not repo_paths:
            raise ValueError("No repository paths provided")
        documents = await self._create_documents(repo_paths)
        if not documents:
            raise NoPythonFilesError("No Python files found in repositories")
        logger.info("Created %d documents", len(documents))
        if augment:
            documents = await self._augment_documents(documents)
        await self._create_index(documents)

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
        # Repository files are third-party text: their tags are defanged.
        return [{"code": neutralize_prompt_tags(doc.page_content)} for doc in documents]

    @staged("code_analysis")
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

    async def _create_index(self, documents: list[Document]) -> None:
        if not documents:
            logger.warning("No documents provided to create index.")
            return

        logger.info(
            "Creating the code index for %d documents in batches of %d.",
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
        logger.info("Indexed %d code chunks.", len(documents))

    @RetryableBase._retry("documents_embedding")
    async def _afrom_documents_with_retry(
        self, documents: list[Document]
    ) -> InMemoryVectorStore:
        return await InMemoryVectorStore.afrom_documents(documents, self.embeddings)

    @RetryableBase._retry("documents_embedding")
    async def _aadd_documents_with_retry(self, documents: list[Document]) -> None:
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")
        await self.vector_store.aadd_documents(documents)

    async def delete_index(self) -> None:
        """Release the in-memory index."""
        self.vector_store = None

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
