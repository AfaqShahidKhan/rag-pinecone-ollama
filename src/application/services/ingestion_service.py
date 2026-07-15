"""
src/application/services/ingestion_service.py

Orchestrates the ingestion pipeline:
    load → pre_process → chunk → embed → upsert

The only change from the original is the addition of the optional
IDocumentProcessor step between load and chunk. When no pre-processor
is injected (pre_processor=None), the pipeline behaves exactly as before
— fully backward compatible.
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


class IngestionService:
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

    def ingest_path(self, source: Path) -> int:
        """
        Ingest a single file or every supported file in a directory.
        Returns the number of vectors upserted.
        """
        documents = self._load(source)
        documents = self._pre_process(documents)
        chunks = self._chunker.chunk(documents)
        embedded_chunks = self._embed(chunks)

        self._vector_store.ensure_index_exists()
        total = self._vector_store.upsert(embedded_chunks)

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