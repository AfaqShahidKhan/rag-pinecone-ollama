"""
src/factories/service_factory.py

Factory that assembles application-layer services from adapters produced
by AdapterFactory. This keeps constructor wiring out of the composition
root's main flow and in one focused, single-responsibility class.
"""

from __future__ import annotations

from typing import Callable

from src.application.services import (
    EvaluationService,
    IngestionService,
    RagQueryService,
    RetrievalService,
)
from src.config.settings import Settings
from src.domain.interfaces import ILogger
from src.factories.adapter_factory import AdapterFactory


class ServiceFactory:
    def __init__(self, settings: Settings, adapter_factory: AdapterFactory, logger_factory: Callable[[str], ILogger]) -> None:
        self._settings = settings
        self._adapters = adapter_factory
        self._logger_factory = logger_factory

    def create_ingestion_service(self) -> IngestionService:
        embedding_provider = self._adapters.create_embedding_provider()
        return IngestionService(
            loader_resolver=self._adapters.create_document_loader_resolver(),
            chunker=self._adapters.create_text_chunker(),
            embedding_provider=embedding_provider,
            vector_store=self._adapters.create_vector_store(embedding_provider.dimension),
            logger=self._logger_factory("ingestion_service"),
        )

    def create_retrieval_service(self) -> RetrievalService:
        embedding_provider = self._adapters.create_embedding_provider()
        return RetrievalService(
            embedding_provider=embedding_provider,
            vector_store=self._adapters.create_vector_store(embedding_provider.dimension),
            logger=self._logger_factory("retrieval_service"),
            retrieval_settings=self._settings.retrieval,
        )

    def create_rag_query_service(self, token_sink: Callable[[str], None] | None = None) -> RagQueryService:
        return RagQueryService(
            retrieval_service=self.create_retrieval_service(),
            prompt_builder=self._adapters.create_prompt_builder(),
            answer_generator=self._adapters.create_answer_generator(token_sink=token_sink),
            logger=self._logger_factory("rag_query_service"),
        )

    def create_evaluation_service(self, token_sink: Callable[[str], None] | None = None) -> EvaluationService:
        return EvaluationService(
            rag_query_service=self.create_rag_query_service(token_sink=token_sink),
            retrieval_service=self.create_retrieval_service(),
            prompt_builder=self._adapters.create_prompt_builder(),
            reporter=self._adapters.create_eval_reporter(),
            logger=self._logger_factory("evaluation_service"),
        )
