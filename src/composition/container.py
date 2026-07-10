"""
src/composition/container.py

Composition root. The only module allowed to call factories.
Accepts an optional vector_store_type override so the Streamlit UI
can switch databases without restarting the process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.application.services import EvaluationService, IngestionService, RagQueryService
from src.config.settings import Settings, VectorStoreType
from src.domain.interfaces import ILogger
from src.factories import AdapterFactory, LoggerFactory, ServiceFactory, SettingsFactory


class Container:
    def __init__(
        self,
        settings: Settings,
        service_factory: ServiceFactory,
        token_sink: Callable[[str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._services = service_factory
        self._token_sink = token_sink
        self._ingestion_service: IngestionService | None = None
        self._rag_query_service: RagQueryService | None = None
        self._evaluation_service: EvaluationService | None = None

    @classmethod
    def bootstrap(
        cls,
        project_root: Path | None = None,
        token_sink: Callable[[str], None] | None = None,
        vector_store_type: VectorStoreType | None = None,
    ) -> "Container":
        root = project_root or Path.cwd()

        settings = SettingsFactory(project_root=root).create(
            vector_store_type=vector_store_type
        )

        logger_factory: Callable[[str], ILogger] = LoggerFactory.create
        adapter_factory = AdapterFactory(
            settings=settings,
            logger_factory=logger_factory,
            vector_store_type=vector_store_type,
        )
        service_factory = ServiceFactory(
            settings=settings,
            adapter_factory=adapter_factory,
            logger_factory=logger_factory,
        )

        return cls(settings=settings, service_factory=service_factory, token_sink=token_sink)

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def ingestion_service(self) -> IngestionService:
        if self._ingestion_service is None:
            self._ingestion_service = self._services.create_ingestion_service()
        return self._ingestion_service

    @property
    def rag_query_service(self) -> RagQueryService:
        if self._rag_query_service is None:
            self._rag_query_service = self._services.create_rag_query_service(
                token_sink=self._token_sink
            )
        return self._rag_query_service

    @property
    def evaluation_service(self) -> EvaluationService:
        if self._evaluation_service is None:
            self._evaluation_service = self._services.create_evaluation_service(
                token_sink=self._token_sink
            )
        return self._evaluation_service