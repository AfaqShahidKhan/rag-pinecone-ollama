"""
src/composition/container.py

Composition root. The only module allowed to call factories.
Exposes every service the UI and CLI need through clean public properties.
No caller should ever access private attributes (_services, _adapters etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.application.services import (
    EvaluationService,
    IngestionService,
    RagQueryService,
    RetrievalService,
    StreamingIngestionService,
)
from src.config.settings import Settings, VectorStoreType
from src.domain.interfaces import ILandingZoneWatcher, ILogger, IRelationalStore
from src.factories import AdapterFactory, LoggerFactory, ServiceFactory, SettingsFactory


class Container:
    def __init__(
        self,
        settings: Settings,
        service_factory: ServiceFactory,
        adapter_factory: AdapterFactory,
        token_sink: Callable[[str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._services = service_factory
        self._adapters = adapter_factory
        self._token_sink = token_sink

        # Lazily-cached services
        self._ingestion_service: IngestionService | None = None
        self._streaming_ingestion_service: StreamingIngestionService | None = None
        self._retrieval_service: RetrievalService | None = None
        self._rag_query_service: RagQueryService | None = None
        self._evaluation_service: EvaluationService | None = None
        self._watcher: ILandingZoneWatcher | None = None
        self._relational_store: IRelationalStore | None = None

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
        return cls(
            settings=settings,
            service_factory=service_factory,
            adapter_factory=adapter_factory,
            token_sink=token_sink,
        )

    # ── Settings ───────────────────────────────────────────────────────────────

    @property
    def settings(self) -> Settings:
        return self._settings

    # ── Ingestion ──────────────────────────────────────────────────────────────

    @property
    def ingestion_service(self) -> IngestionService:
        if self._ingestion_service is None:
            self._ingestion_service = self._services.create_ingestion_service()
        return self._ingestion_service

    @property
    def streaming_ingestion_service(self) -> StreamingIngestionService:
        if self._streaming_ingestion_service is None:
            self._streaming_ingestion_service = self._services.create_streaming_ingestion_service()
        return self._streaming_ingestion_service

    # ── Retrieval & RAG ────────────────────────────────────────────────────────

    @property
    def retrieval_service(self) -> RetrievalService:
        """Public access to retrieval — used by the Debug tab."""
        if self._retrieval_service is None:
            self._retrieval_service = self._services.create_retrieval_service()
        return self._retrieval_service

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

    # ── Landing zone ───────────────────────────────────────────────────────────

    @property
    def watcher(self) -> ILandingZoneWatcher:
        if self._watcher is None:
            self._watcher = self._services.create_landing_zone_watcher()
        return self._watcher

    def create_watcher(self, recursive: bool = False) -> ILandingZoneWatcher:
        """Create a fresh watcher — used by the Watch tab so recursive flag can vary."""
        return self._services.create_landing_zone_watcher(recursive=recursive)

    # ── Relational store ───────────────────────────────────────────────────────

    @property
    def relational_store(self) -> IRelationalStore | None:
        """
        Returns None when RELATIONAL_STORE_ENABLED=false.
        The Relational Store tab checks for None before rendering.
        """
        if self._relational_store is None:
            self._relational_store = self._adapters.create_relational_store()
        return self._relational_store

    # ── Prompt builder (needed by Debug tab) ──────────────────────────────────

    def build_prompt(self, question: str, results):
        """Convenience method so the Debug tab never accesses private attributes."""
        builder = self._adapters.create_prompt_builder()
        return builder.build(question, results)