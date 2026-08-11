"""
src/application/services/corpus_builder_service.py

Runs only the document-parsing portion of the ingestion pipeline:
    [validate file] → load → [extract images] → [validate content]
    → pre_process → write_corpus

Deliberately does NOT depend on IEmbeddingProvider or IVectorStore — this
service works with zero external service credentials configured.

Validation step: same FileValidationGate as IngestionService/
StreamingIngestionService — read-only check, empty-file check, and content
validators all apply here too, so build-corpus is a true preview of what
the full ingest would accept or reject.
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
from src.application.services.file_validation_gate import FileValidationGate


class CorpusBuilderService:
    def __init__(
        self,
        loader_resolver: IDocumentLoaderResolver,
        logger: ILogger,
        pre_processor: IDocumentProcessor | None = None,
        image_extractor_resolver: IImageExtractorResolver | None = None,
        corpus_writer: ICorpusWriter | None = None,
        validation_gate: FileValidationGate | None = None,
    ) -> None:
        self._loader_resolver = loader_resolver
        self._logger = logger
        self._pre_processor = pre_processor
        self._image_extractor_resolver = image_extractor_resolver
        self._corpus_writer = corpus_writer
        self._validation_gate = validation_gate

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