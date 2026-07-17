"""
src/factories/adapter_factory.py

Abstract factory that constructs every infrastructure adapter.
The only place in the codebase where concrete adapter classes are instantiated.
create_vector_store() selects the correct adapter based on VectorStoreType.
"""

from __future__ import annotations

from typing import Callable

from rich.console import Console

from src.config.settings import Settings, VectorStoreType
from src.domain.interfaces import (
    IAnswerGenerator,
    IDocumentLoader,
    IEmbeddingProvider,
    IEvalReporter,
    ILogger,
    IPromptBuilder,
    ITextChunker,
    IVectorIdStrategy,
    IVectorStore,
)
from src.factories.document_loader_factory import DocumentLoaderFactory
from src.factories.sdk_client_factory import SdkClientFactory
from src.infrastructure.chunking import RecursiveTextChunker
from src.infrastructure.embeddings import OllamaEmbeddingProvider
from src.infrastructure.generation import DefaultPromptBuilder, OllamaAnswerGenerator
from src.infrastructure.loaders import DocxDocumentLoader, PdfDocumentLoader
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
        # UI override wins; falls back to whatever is in Settings
        self._vector_store_type = vector_store_type or settings.vector_store_type

    def create_document_loaders(self) -> list[IDocumentLoader]:
        return [
            PdfDocumentLoader(logger=self._logger_factory("loaders.pdf")),
            DocxDocumentLoader(
                logger=self._logger_factory("loaders.docx"),
                ingestion_settings=self._settings.ingestion,
            ),
        ]

    def create_document_loader_resolver(self) -> DocumentLoaderFactory:
        return DocumentLoaderFactory(
            loaders=self.create_document_loaders(),
            logger=self._logger_factory("loaders.resolver"),
        )

    def create_text_chunker(self) -> ITextChunker:
        return RecursiveTextChunker(
            logger=self._logger_factory("chunking"),
            chunking_settings=self._settings.chunking,
        )

    def create_embedding_provider(self) -> IEmbeddingProvider:
        ollama_client = SdkClientFactory.create_ollama_client(self._settings.ollama)
        return OllamaEmbeddingProvider(
            client=ollama_client,
            logger=self._logger_factory("embeddings"),
            ollama_settings=self._settings.ollama,
            ingestion_settings=self._settings.ingestion,
        )

    def create_vector_id_strategy(self) -> IVectorIdStrategy:
        return Sha256VectorIdStrategy()

    def create_vector_store(self, embedding_dimension: int) -> IVectorStore:
        """Select and build the correct vector store adapter based on VectorStoreType."""
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

    def create_eval_reporter(self, console: Console | None = None) -> IEvalReporter:
        return RichEvalReporter(console=console or Console())