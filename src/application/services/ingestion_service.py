"""
src/application/services/ingestion_service.py

Orchestrates the ingestion pipeline:
    load → pre_process → chunk → embed → upsert → [save to relational store]

Phase 5: optionally persists embedded chunks to the relational store
after the vector store upsert, enabling parent-child retrieval.
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


class IngestionService:
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

    def ingest_path(self, source: Path) -> int:
        documents = self._load(source)
        documents = self._pre_process(documents)
        chunks = self._chunker.chunk(documents)
        embedded_chunks = self._embed(chunks)

        self._vector_store.ensure_index_exists()
        total = self._vector_store.upsert(embedded_chunks)

        # Phase 5: persist to relational store if enabled
        if self._relational_store and self._id_strategy:
            self._relational_store.ensure_schema()
            vector_ids = [self._id_strategy.generate_id(c) for c in embedded_chunks]
            self._relational_store.save_chunks(embedded_chunks, vector_ids)

        self._logger.info(
            f"Pipeline complete — {total} vectors indexed from '{source.name}'."
        )
        return total

    def _load(self, source: Path) -> list[Document]:
        if source.is_file():
            loader = self._loader_resolver.resolve_for_file(source)
            return loader.load(source)
        return self._loader_resolver.load_all_from_directory(source)

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