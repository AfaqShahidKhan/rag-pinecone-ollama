"""
src/application/services/streaming_ingestion_service.py

Processes documents one file at a time through the full ingestion pipeline:
    load → pre_process → chunk → embed → upsert

Key difference from IngestionService:
  - IngestionService (batch mode): loads ALL files first, then processes
    the entire corpus as one batch. Fast but holds all pages in memory.
  - StreamingIngestionService (streaming mode): processes each file
    independently and completely before moving to the next. Peak memory
    stays proportional to the largest single file, not the entire corpus.

This is what FileIngestionAdapter calls for event-driven ingestion
(landing zone watcher mode), and can also be called directly via CLI
with `python main.py watch`.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.entities import Document, EmbeddedChunk
from src.domain.interfaces import (
    IDocumentLoaderResolver,
    IDocumentProcessor,
    IEmbeddingProvider,
    ILogger,
    ITextChunker,
    IVectorStore,
)


class StreamingIngestionService:
    def __init__(
        self,
        loader_resolver: IDocumentLoaderResolver,
        chunker: ITextChunker,
        embedding_provider: IEmbeddingProvider,
        vector_store: IVectorStore,
        logger: ILogger,
        pre_processor: IDocumentProcessor | None = None,
    ) -> None:
        self._loader_resolver = loader_resolver
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._logger = logger
        self._pre_processor = pre_processor
        self._index_ready = False

    def ingest_file(self, path: Path) -> int:
        """
        Run a single file through the complete pipeline.
        Returns the number of vectors upserted.
        Raises ValueError for unsupported file types.
        """
        loader = self._loader_resolver.resolve_for_file(path)
        documents = loader.load(path)

        if not documents:
            self._logger.warning(
                f"StreamingIngestionService: '{path.name}' produced no documents."
            )
            return 0

        documents = self._pre_process(documents)
        if not documents:
            self._logger.warning(
                f"StreamingIngestionService: '{path.name}' produced no documents "
                f"after pre-processing."
            )
            return 0

        chunks = self._chunker.chunk(documents)
        if not chunks:
            return 0

        embedded = self._embed(chunks)
        self._ensure_index()
        total = self._vector_store.upsert(embedded)

        self._logger.info(
            f"StreamingIngestionService: '{path.name}' — {total} vectors indexed."
        )
        return total

    def ingest_directory(self, directory: Path) -> int:
        """
        Process every supported file in a directory independently,
        one file at a time. Returns total vectors upserted.
        """
        files = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and self._is_supported(p)
        )

        if not files:
            raise FileNotFoundError(
                f"No supported files found in '{directory}'."
            )

        self._logger.info(
            f"StreamingIngestionService: processing {len(files)} file(s) "
            f"from '{directory}' one at a time."
        )

        total = 0
        for file_path in files:
            try:
                total += self.ingest_file(file_path)
            except ValueError as exc:
                self._logger.warning(f"Skipping '{file_path.name}': {exc}")

        self._logger.info(
            f"StreamingIngestionService: complete — {total} total vectors indexed."
        )
        return total

    def _pre_process(self, documents: list[Document]) -> list[Document]:
        if self._pre_processor is None:
            return documents
        return self._pre_processor.process_all(documents)

    def _embed(self, chunks: list[Document]) -> list[EmbeddedChunk]:
        texts = [doc.page_content for doc in chunks]
        vectors = self._embedding_provider.embed_texts(texts)
        return [
            EmbeddedChunk(document=doc, vector=vec)
            for doc, vec in zip(chunks, vectors)
        ]

    def _ensure_index(self) -> None:
        """Call ensure_index_exists() only once per service lifetime."""
        if not self._index_ready:
            self._vector_store.ensure_index_exists()
            self._index_ready = True

    def _is_supported(self, path: Path) -> bool:
        try:
            self._loader_resolver.resolve_for_file(path)
            return True
        except ValueError:
            return False