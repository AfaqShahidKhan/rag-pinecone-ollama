"""
src/factories/adapter_factory.py

Abstract factory that constructs every infrastructure adapter.
Phase 5: adds create_pii_pre_processor() and create_relational_store().
The pre-processing pipeline now conditionally includes PII redaction.
"""

from __future__ import annotations

import os
from typing import Callable

from rich.console import Console

from src.config.settings import Settings, VectorStoreType
from src.domain.interfaces import (
    IAnswerGenerator,
    IDocumentLoader,
    IDocumentProcessor,
    IEmbeddingProvider,
    IEvalReporter,
    IIngestionAdapter,
    ILandingZoneWatcher,
    ILogger,
    IPromptBuilder,
    IRelationalStore,
    ITextChunker,
    IVectorIdStrategy,
    IVectorStore,
)
from src.factories.document_loader_factory import DocumentLoaderFactory
from src.factories.sdk_client_factory import SdkClientFactory
from src.infrastructure.chunking import (
    ChunkingRoute,
    DocumentRouter,
    RecursiveTextChunker,
    SemanticChunker,
)
from src.infrastructure.embeddings import OllamaEmbeddingProvider
from src.infrastructure.generation import DefaultPromptBuilder, OllamaAnswerGenerator
from src.infrastructure.landing_zone import FileIngestionAdapter, FileSystemWatcher
from src.infrastructure.loaders import (
    DocxDocumentLoader,
    HtmlLoader,
    JsonLoader,
    OcrLoader,
    PdfDocumentLoader,
)
from src.infrastructure.pii import RegexPiiAnonymizer
from src.infrastructure.pre_processing import (
    MetadataEnricher,
    MetadataNormalizer,
    PiiAnonymizingPreProcessor,
    PreProcessingPipeline,
    SchemaMapper,
    TextSanitizer,
    UnicodeNormalizer,
)
from src.infrastructure.relational_store import SqliteRelationalStore
from src.infrastructure.reporting import RichEvalReporter
from src.infrastructure.vector_store import (
    ChromaVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    Sha256VectorIdStrategy,
)


class AdapterFactory:
    def __init__(
        self,
        settings: Settings,
        logger_factory: Callable[[str], ILogger],
        vector_store_type: VectorStoreType | None = None,
    ) -> None:
        self._settings = settings
        self._logger_factory = logger_factory
        self._vector_store_type = vector_store_type or settings.vector_store_type

    # ── Document loading ───────────────────────────────────────────────────────

    def create_document_loaders(self) -> list[IDocumentLoader]:
        ocr_lang = os.getenv("TESSERACT_LANG", "eng")
        return [
            PdfDocumentLoader(logger=self._logger_factory("loaders.pdf")),
            DocxDocumentLoader(
                logger=self._logger_factory("loaders.docx"),
                ingestion_settings=self._settings.ingestion,
            ),
            HtmlLoader(logger=self._logger_factory("loaders.html")),
            JsonLoader(logger=self._logger_factory("loaders.json")),
            OcrLoader(logger=self._logger_factory("loaders.ocr"), lang=ocr_lang),
        ]

    def create_document_loader_resolver(self) -> DocumentLoaderFactory:
        return DocumentLoaderFactory(
            loaders=self.create_document_loaders(),
            logger=self._logger_factory("loaders.resolver"),
        )

    # ── Pre-processing (Phase 1 + 3 + 5) ──────────────────────────────────────

    def create_pre_processing_pipeline(self) -> IDocumentProcessor:
        """
        Full pre-processing chain:
          TextSanitizer → UnicodeNormalizer → MetadataNormalizer
          → SchemaMapper → MetadataEnricher → [PiiAnonymizer if enabled]
        """
        processors = [
            TextSanitizer(logger=self._logger_factory("pre_processing.sanitizer")),
            UnicodeNormalizer(logger=self._logger_factory("pre_processing.unicode")),
            MetadataNormalizer(logger=self._logger_factory("pre_processing.metadata")),
            SchemaMapper(logger=self._logger_factory("pre_processing.schema")),
            MetadataEnricher(logger=self._logger_factory("pre_processing.enricher")),
        ]

        # Conditionally add PII anonymizer (Phase 5)
        if self._settings.pii.enabled:
            enabled_types = list(self._settings.pii.enabled_types) or None
            anonymizer = RegexPiiAnonymizer(
                logger=self._logger_factory("pii.anonymizer"),
                enabled_types=enabled_types,
            )
            processors.append(
                PiiAnonymizingPreProcessor(
                    anonymizer=anonymizer,
                    logger=self._logger_factory("pre_processing.pii"),
                )
            )

        return PreProcessingPipeline(
            processors=processors,
            logger=self._logger_factory("pre_processing.pipeline"),
        )

    # ── Chunking ───────────────────────────────────────────────────────────────

    def create_text_chunker(self) -> ITextChunker:
        recursive_chunker = RecursiveTextChunker(
            logger=self._logger_factory("chunking.recursive"),
            chunking_settings=self._settings.chunking,
        )
        ollama_client = SdkClientFactory.create_ollama_client(self._settings.ollama)
        embedding_provider = OllamaEmbeddingProvider(
            client=ollama_client,
            logger=self._logger_factory("chunking.semantic.embedder"),
            ollama_settings=self._settings.ollama,
            ingestion_settings=self._settings.ingestion,
        )
        semantic_chunker = SemanticChunker(
            embedding_provider=embedding_provider,
            logger=self._logger_factory("chunking.semantic"),
            semantic_settings=self._settings.semantic_chunking,
        )
        routes = [
            ChunkingRoute(
                name="semantic",
                file_types=frozenset({"html", "json"}),
                chunker=semantic_chunker,
            ),
        ]
        return DocumentRouter(
            routes=routes,
            default_chunker=recursive_chunker,
            logger=self._logger_factory("chunking.router"),
        )

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def create_embedding_provider(self) -> IEmbeddingProvider:
        ollama_client = SdkClientFactory.create_ollama_client(self._settings.ollama)
        return OllamaEmbeddingProvider(
            client=ollama_client,
            logger=self._logger_factory("embeddings"),
            ollama_settings=self._settings.ollama,
            ingestion_settings=self._settings.ingestion,
        )

    # ── Vector store ───────────────────────────────────────────────────────────

    def create_vector_id_strategy(self) -> IVectorIdStrategy:
        return Sha256VectorIdStrategy()

    def create_vector_store(self, embedding_dimension: int) -> IVectorStore:
        vst = self._vector_store_type
        id_strategy = self.create_vector_id_strategy()
        logger = self._logger_factory("vector_store")

        if vst == VectorStoreType.PINECONE:
            pinecone_client = SdkClientFactory.create_pinecone_client(self._settings.pinecone)
            return PineconeVectorStore(
                client=pinecone_client,
                id_strategy=id_strategy,
                logger=logger,
                pinecone_settings=self._settings.pinecone,
                ingestion_settings=self._settings.ingestion,
                embedding_dimension=embedding_dimension,
            )
        if vst == VectorStoreType.CHROMA:
            return ChromaVectorStore(
                id_strategy=id_strategy,
                logger=logger,
                chroma_settings=self._settings.chroma,
                ingestion_settings=self._settings.ingestion,
                embedding_dimension=embedding_dimension,
            )
        if vst == VectorStoreType.QDRANT:
            return QdrantVectorStore(
                id_strategy=id_strategy,
                logger=logger,
                qdrant_settings=self._settings.qdrant,
                ingestion_settings=self._settings.ingestion,
                embedding_dimension=embedding_dimension,
            )
        raise ValueError(f"Unsupported VectorStoreType: {vst}")

    # ── Relational store (Phase 5) ─────────────────────────────────────────────

    def create_relational_store(self) -> IRelationalStore | None:
        """Returns None when RELATIONAL_STORE_ENABLED=false in .env."""
        if not self._settings.relational_store.enabled:
            return None
        return SqliteRelationalStore(
            logger=self._logger_factory("relational_store"),
            settings=self._settings.relational_store,
        )

    # ── Generation ─────────────────────────────────────────────────────────────

    def create_prompt_builder(self) -> IPromptBuilder:
        return DefaultPromptBuilder(
            logger=self._logger_factory("prompt_builder"),
            prompt_settings=self._settings.prompt,
        )

    def create_answer_generator(
        self, token_sink: Callable[[str], None] | None = None
    ) -> IAnswerGenerator:
        ollama_client = SdkClientFactory.create_ollama_client(self._settings.ollama)
        return OllamaAnswerGenerator(
            client=ollama_client,
            logger=self._logger_factory("generator"),
            ollama_settings=self._settings.ollama,
            token_sink=token_sink,
        )

    # ── Reporting ──────────────────────────────────────────────────────────────

    def create_eval_reporter(self, console: Console | None = None) -> IEvalReporter:
        return RichEvalReporter(console=console or Console())

    # ── Landing zone ───────────────────────────────────────────────────────────

    def create_file_ingestion_adapter(self, streaming_service) -> IIngestionAdapter:
        return FileIngestionAdapter(
            ingestion_service=streaming_service,
            logger=self._logger_factory("landing_zone.adapter"),
        )

    def create_file_system_watcher(
        self, adapter: IIngestionAdapter, recursive: bool = False
    ) -> ILandingZoneWatcher:
        return FileSystemWatcher(
            adapter=adapter,
            logger=self._logger_factory("landing_zone.watcher"),
            recursive=recursive,
        )