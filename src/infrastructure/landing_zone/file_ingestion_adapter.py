"""
src/infrastructure/landing_zone/file_ingestion_adapter.py

IIngestionAdapter adapter that receives a file path from the watcher and
delegates to StreamingIngestionService for single-file processing.

Single responsibility: translate a raw file path into a structured,
logged ingestion call. It does not know about OS events, watchdog,
or chunking — those concerns belong to other layers.
"""

from __future__ import annotations

from pathlib import Path

from src.application.services.streaming_ingestion_service import StreamingIngestionService
from src.domain.interfaces import IIngestionAdapter, ILogger


class FileIngestionAdapter(IIngestionAdapter):
    def __init__(
        self,
        ingestion_service: StreamingIngestionService,
        logger: ILogger,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._logger = logger

    def on_file_arrived(self, path: Path) -> None:
        self._logger.info(
            f"FileIngestionAdapter: ingesting '{path.name}'..."
        )
        try:
            total = self._ingestion_service.ingest_file(path)
            self._logger.info(
                f"FileIngestionAdapter: '{path.name}' ingested — "
                f"{total} vectors indexed."
            )
        except ValueError as exc:
            # Unsupported file type — log and skip, do not crash the watcher
            self._logger.warning(
                f"FileIngestionAdapter: skipping '{path.name}' — {exc}"
            )
        except Exception as exc:
            self._logger.error(
                f"FileIngestionAdapter: failed to ingest '{path.name}' — {exc}"
            )