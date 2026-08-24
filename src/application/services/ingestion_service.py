"""
src/application/services/ingestion_service.py

Orchestrates the ingestion pipeline:
    [validate file] → load → [extract images] → [validate content]
    → pre_process → [write corpus] → chunk → embed → upsert
    → [save to relational store]

Validation step: a FileValidationGate runs before AND after loading each
file. Rejected files (writable/not-readonly, empty, load failure, empty or
corrupt content) are logged with an error code and moved to
data/unprocessed/<reason>/ — they never reach chunking/embedding.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.entities import Document, EmbeddedChunk
from src.domain.interfaces import (
    ICorpusWriter,
    IDocumentLoaderResolver,
    IDocumentProcessor,
    IEmbeddingProvider,
    IImageExtractorResolver,
    ILogger,
    IRelationalStore,
    ITextChunker,
    IVectorIdStrategy,
    IVectorStore,
)
from src.application.services.file_validation_gate import FileValidationGate


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
        corpus_writer: ICorpusWriter | None = None,
        image_extractor_resolver: IImageExtractorResolver | None = None,
        validation_gate: FileValidationGate | None = None,
    ) -> None:
        self._loader_resolver = loader_resolver
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._logger = logger
        self._pre_processor = pre_processor
        self._relational_store = relational_store
        self._id_strategy = id_strategy
        self._corpus_writer = corpus_writer
        self._image_extractor_resolver = image_extractor_resolver
        self._validation_gate = validation_gate

    def ingest_path(self, source: Path) -> int:
        documents = self._load(source)
        documents = self._extract_images(documents)
        documents = self._pre_process(documents)
        self._write_corpus(documents)
        chunks = self._chunker.chunk(documents)
        embedded_chunks = self._embed(chunks)

        self._vector_store.ensure_index_exists()
        total = self._vector_store.upsert(embedded_chunks)

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
            return self._load_single_file(source)

        files = self._loader_resolver.list_supported_files(source)
        all_docs: list[Document] = []
        for path in files:
            all_docs.extend(self._load_single_file(path))
        return all_docs

    def _load_single_file(self, path: Path) -> list[Document]:
        if self._validation_gate and not self._validation_gate.check_file(path):
            return []

        loader = self._loader_resolver.resolve_for_file(path)
        try:
            documents = loader.load(path)
        except Exception as exc:
            if self._validation_gate:
                self._validation_gate.reject_load_exception(path, exc)
            else:
                self._logger.error(f"Failed to load '{path.name}': {exc}")
            return []

        if self._validation_gate and not self._validation_gate.check_content(path, documents):
            return []

        return documents

    def _extract_images(self, documents: list[Document]) -> list[Document]:
        if self._image_extractor_resolver is None:
            return documents

        seen_paths: dict[str, Path] = {}
        for doc in documents:
            source_path = doc.metadata.get("source_path")
            if source_path and source_path not in seen_paths:
                seen_paths[source_path] = Path(source_path)

        for source_path, path in seen_paths.items():
            extractor = self._image_extractor_resolver.resolve_for_file(path)
            if extractor is None:
                continue
            documents = extractor.extract(path, documents)

        return documents

    def _pre_process(self, documents: list[Document]) -> list[Document]:
        if self._pre_processor is None:
            return documents
        return self._pre_processor.process_all(documents)

    def _write_corpus(self, documents: list[Document]) -> None:
        if self._corpus_writer is None:
            return
        self._corpus_writer.write(documents)

    def _embed(self, chunks: list[Document]) -> list[EmbeddedChunk]:
        texts = [doc.page_content for doc in chunks]
        vectors = self._embedding_provider.embed_texts(texts)
        return [
            EmbeddedChunk(document=doc, vector=vec)
            for doc, vec in zip(chunks, vectors)
        ]