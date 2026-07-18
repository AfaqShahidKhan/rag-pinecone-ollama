"""
src/application/services/streaming_ingestion_service.py

Processes documents one file at a time.
Phase 5: optionally persists embedded chunks to the relational store.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.entities import Document, EmbeddedChunk
from src.domain.interfaces import (
    IDocumentLoaderResolver,
    IDocumentProcessor,
    IEmbeddingProvider,
    ILogger,
    IRelationalStore,
    ITextChunker,
    IVectorIdStrategy,
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
        relational_store: IRelationalStore | None = None,
        id_strategy: IVectorIdStrategy | None = None,
    ) -> None:
        self._loader_resolver = loader_resolver
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._logger = logger
        self._pre_processor = pre_processor
        self._relational_store = relational_store
        self._id_strategy = id_strategy
        self._index_ready = False

    def ingest_file(self, path: Path) -> int:
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
                "after pre-processing."
            )
            return 0

        chunks = self._chunker.chunk(documents)
        if not chunks:
            return 0

        embedded = self._embed(chunks)
        self._ensure_index()
        total = self._vector_store.upsert(embedded)

        # Phase 5: persist to relational store if enabled
        if self._relational_store and self._id_strategy:
            self._relational_store.ensure_schema()
            vector_ids = [self._id_strategy.generate_id(c) for c in embedded]
            self._relational_store.save_chunks(embedded, vector_ids)

        self._logger.info(
            f"StreamingIngestionService: '{path.name}' — {total} vectors indexed."
        )
        return total

    def ingest_directory(self, directory: Path) -> int:
        files = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and self._is_supported(p)
        )
        if not files:
            raise FileNotFoundError(f"No supported files found in '{directory}'.")

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
        if not self._index_ready:
            self._vector_store.ensure_index_exists()
            self._index_ready = True

    def _is_supported(self, path: Path) -> bool:
        try:
            self._loader_resolver.resolve_for_file(path)
            return True
        except ValueError:
            return False