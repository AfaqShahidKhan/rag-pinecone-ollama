"""
src/factories/service_factory.py

Assembles application-layer services from adapters.
Phase 5: injects relational_store and id_strategy into both ingestion services.
Corpus step: injects corpus_writer into both ingestion services.
Image/table step: injects image_extractor_resolver into both ingestion services.
Test data step: adds create_corpus_builder_service().
Validation step: adds create_file_validation_gate(), injected into
IngestionService, StreamingIngestionService, and CorpusBuilderService — the
read-only check, structural validators, and unprocessed/ quarantine now
behave identically across ingest/build-corpus/watch.
"""

from __future__ import annotations

from typing import Callable

from src.application.services import (
    CorpusBuilderService,
    EvaluationService,
    FileValidationGate,
    IngestionService,
    RagQueryService,
    RetrievalService,
    StreamingIngestionService,
)
from src.config.settings import Settings
from src.domain.interfaces import ILandingZoneWatcher, ILogger
from src.factories.adapter_factory import AdapterFactory


class ServiceFactory:
    def __init__(
        self,
        settings: Settings,
        adapter_factory: AdapterFactory,
        logger_factory: Callable[[str], ILogger],
    ) -> None:
        self._settings = settings
        self._adapters = adapter_factory
        self._logger_factory = logger_factory

    def create_file_validation_gate(self) -> FileValidationGate | None:
        """Returns None when VALIDATION_ENABLED=false — services then skip all gating."""
        if not self._settings.validation.enabled:
            return None
        return FileValidationGate(
            file_validators=self._adapters.create_file_validators(),
            content_validators=self._adapters.create_content_validators(),
            mover=self._adapters.create_unprocessed_mover(),
            logger=self._logger_factory("validation.gate"),
        )

    def create_ingestion_service(self) -> IngestionService:
        embedding_provider = self._adapters.create_embedding_provider()
        return IngestionService(
            loader_resolver=self._adapters.create_document_loader_resolver(),
            chunker=self._adapters.create_text_chunker(),
            embedding_provider=embedding_provider,
            vector_store=self._adapters.create_vector_store(embedding_provider.dimension),
            logger=self._logger_factory("ingestion_service"),
            pre_processor=self._adapters.create_pre_processing_pipeline(),
            relational_store=self._adapters.create_relational_store(),
            id_strategy=self._adapters.create_vector_id_strategy(),
            corpus_writer=self._adapters.create_corpus_writer(),
            image_extractor_resolver=self._adapters.create_image_extractor_resolver(),
            validation_gate=self.create_file_validation_gate(),
        )

    def create_streaming_ingestion_service(self) -> StreamingIngestionService:
        embedding_provider = self._adapters.create_embedding_provider()
        return StreamingIngestionService(
            loader_resolver=self._adapters.create_document_loader_resolver(),
            chunker=self._adapters.create_text_chunker(),
            embedding_provider=embedding_provider,
            vector_store=self._adapters.create_vector_store(embedding_provider.dimension),
            logger=self._logger_factory("streaming_ingestion_service"),
            pre_processor=self._adapters.create_pre_processing_pipeline(),
            relational_store=self._adapters.create_relational_store(),
            id_strategy=self._adapters.create_vector_id_strategy(),
            corpus_writer=self._adapters.create_corpus_writer(),
            image_extractor_resolver=self._adapters.create_image_extractor_resolver(),
            validation_gate=self.create_file_validation_gate(),
        )

    def create_corpus_builder_service(self) -> CorpusBuilderService:
        """
        Deliberately never calls create_embedding_provider() or
        create_vector_store() — this is what makes it usable with zero
        external service credentials configured.
        """
        return CorpusBuilderService(
            loader_resolver=self._adapters.create_document_loader_resolver(),
            logger=self._logger_factory("corpus_builder_service"),
            pre_processor=self._adapters.create_pre_processing_pipeline(),
            image_extractor_resolver=self._adapters.create_image_extractor_resolver(),
            corpus_writer=self._adapters.create_corpus_writer(),
            validation_gate=self.create_file_validation_gate(),
        )

    def create_landing_zone_watcher(self, recursive: bool = False) -> ILandingZoneWatcher:
        streaming_service = self.create_streaming_ingestion_service()
        adapter = self._adapters.create_file_ingestion_adapter(streaming_service)
        return self._adapters.create_file_system_watcher(
            adapter=adapter,
            recursive=recursive,
        )

    def create_retrieval_service(self) -> RetrievalService:
        embedding_provider = self._adapters.create_embedding_provider()
        return RetrievalService(
            embedding_provider=embedding_provider,
            vector_store=self._adapters.create_vector_store(embedding_provider.dimension),
            logger=self._logger_factory("retrieval_service"),
            retrieval_settings=self._settings.retrieval,
        )

    def create_rag_query_service(
        self, token_sink: Callable[[str], None] | None = None
    ) -> RagQueryService:
        return RagQueryService(
            retrieval_service=self.create_retrieval_service(),
            prompt_builder=self._adapters.create_prompt_builder(),
            answer_generator=self._adapters.create_answer_generator(token_sink=token_sink),
            logger=self._logger_factory("rag_query_service"),
        )

    def create_evaluation_service(
        self, token_sink: Callable[[str], None] | None = None
    ) -> EvaluationService:
        return EvaluationService(
            rag_query_service=self.create_rag_query_service(token_sink=token_sink),
            retrieval_service=self.create_retrieval_service(),
            prompt_builder=self._adapters.create_prompt_builder(),
            reporter=self._adapters.create_eval_reporter(),
            logger=self._logger_factory("evaluation_service"),
        )