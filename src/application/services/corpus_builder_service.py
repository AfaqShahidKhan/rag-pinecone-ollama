"""
src/application/services/corpus_builder_service.py

Runs only the document-parsing portion of the ingestion pipeline:
    load -> extract_images -> pre_process -> write_corpus

Deliberately does NOT depend on IEmbeddingProvider or IVectorStore — unlike
IngestionService, which needs both. That means building this service never
constructs an Ollama or Pinecone/Chroma/Qdrant client, so it works with zero
external service credentials configured. Useful for validating document
parsing, table extraction, image extraction, and PII redaction without
needing Ollama running or a vector store connection at all — exactly what a
teammate testing "landing zone -> corpus" needs.

IngestionService covers the full pipeline (this stage + chunk/embed/upsert);
this service exists so testing the parsing stage doesn't require standing
up everything downstream of it.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.entities import Document
from src.domain.interfaces import (
    ICorpusWriter,
    IDocumentLoaderResolver,
    IDocumentProcessor,
    IImageExtractorResolver,
    ILogger,
)


class CorpusBuilderService:
    def __init__(
        self,
        loader_resolver: IDocumentLoaderResolver,
        logger: ILogger,
        pre_processor: IDocumentProcessor | None = None,
        image_extractor_resolver: IImageExtractorResolver | None = None,
        corpus_writer: ICorpusWriter | None = None,
    ) -> None:
        self._loader_resolver = loader_resolver
        self._logger = logger
        self._pre_processor = pre_processor
        self._image_extractor_resolver = image_extractor_resolver
        self._corpus_writer = corpus_writer

    def build(self, source: Path) -> int:
        if self._corpus_writer is None:
            raise RuntimeError(
                "CorpusBuilderService requires the corpus writer to be enabled "
                "(CORPUS_WRITER_ENABLED=true / corpus.enabled: true in YAML) — "
                "it's currently disabled, so there would be nothing to inspect."
            )

        documents = self._load(source)
        documents = self._extract_images(documents)
        documents = self._pre_process(documents)
        written = self._corpus_writer.write(documents)

        self._logger.info(
            f"Corpus-only build complete — {written} corpus file(s) written from "
            f"'{source.name}' (no embedding or vector store involved)."
        )
        return written

    def _load(self, source: Path) -> list[Document]:
        if source.is_file():
            loader = self._loader_resolver.resolve_for_file(source)
            return loader.load(source)
        return self._loader_resolver.load_all_from_directory(source)

    def _extract_images(self, documents: list[Document]) -> list[Document]:
        if self._image_extractor_resolver is None:
            return documents

        # Group by source file so each file's images are extracted once,
        # even though a directory build produces a flat list across files.
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