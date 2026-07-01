"""
src/composition/container.py

The composition root. This is the ONLY module in the entire codebase that
is allowed to call factories to build concrete services. Every other layer
(application, infrastructure adapters, domain) only ever receives objects
through constructor injection - never constructs its own collaborators.

Usage:
    container = Container.bootstrap()
    container.ingestion_service.ingest_path(Path("data/raw"))
    response = container.rag_query_service.ask("What is the Magi story about?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.application.services import EvaluationService, IngestionService, RagQueryService
from src.config.settings import Settings
from src.domain.interfaces import ILogger
from src.factories import AdapterFactory, LoggerFactory, ServiceFactory, SettingsFactory


class Container:
    """
    Lazily builds and caches the top-level application services.
    Each service is rebuilt from settings only once, the first time it's
    requested, then reused for the lifetime of the container.
    """

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
    def bootstrap(cls, project_root: Path | None = None, token_sink: Callable[[str], None] | None = None) -> "Container":
        root = project_root or Path.cwd()
        settings = SettingsFactory(project_root=root).create()

        logger_factory: Callable[[str], ILogger] = LoggerFactory.create
        adapter_factory = AdapterFactory(settings=settings, logger_factory=logger_factory)
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
            self._rag_query_service = self._services.create_rag_query_service(token_sink=self._token_sink)
        return self._rag_query_service

    @property
    def evaluation_service(self) -> EvaluationService:
        if self._evaluation_service is None:
            self._evaluation_service = self._services.create_evaluation_service(token_sink=self._token_sink)
        return self._evaluation_service
