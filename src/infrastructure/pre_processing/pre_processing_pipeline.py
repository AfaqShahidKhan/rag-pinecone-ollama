"""
src/infrastructure/pre_processing/pre_processing_pipeline.py

Implements IDocumentProcessor by chaining a sequence of IPreProcessors
using the Pipeline Pattern.

Each processor in the chain runs in order on every document. If a
processor raises ValueError (e.g. SchemaMapper rejecting an empty page),
the document is excluded from the output and the error is logged as a
warning — the pipeline never crashes on a single bad document.

Designed to be stateless and re-entrant: the same instance can be called
multiple times across different ingestion runs.
"""

from __future__ import annotations

from src.domain.entities import Document
from src.domain.interfaces import IDocumentProcessor, ILogger, IPreProcessor


class PreProcessingPipeline(IDocumentProcessor):
    def __init__(
        self,
        processors: list[IPreProcessor],
        logger: ILogger,
    ) -> None:
        """
        Args:
            processors: Ordered list of IPreProcessor adapters. They are
                        applied left-to-right to each document.
            logger:     Injected logger — no direct rich/logging import here.
        """
        self._processors = processors
        self._logger = logger

    def process_all(self, documents: list[Document]) -> list[Document]:
        total_in = len(documents)
        results: list[Document] = []

        for doc in documents:
            processed = self._run_processors(doc)
            if processed is not None:
                results.append(processed)

        dropped = total_in - len(results)
        if dropped:
            self._logger.warning(
                f"PreProcessingPipeline: dropped {dropped}/{total_in} documents "
                f"(failed validation). {len(results)} documents passed."
            )
        else:
            self._logger.info(
                f"PreProcessingPipeline: {len(results)} documents processed "
                f"through {len(self._processors)} stages."
            )

        return results

    def _run_processors(self, document: Document) -> Document | None:
        """Apply all processors in order. Return None if any processor rejects the document."""
        current = document
        for processor in self._processors:
            try:
                current = processor.process(current)
            except ValueError as exc:
                self._logger.warning(
                    f"PreProcessingPipeline: document dropped at "
                    f"'{type(processor).__name__}' — {exc}"
                )
                return None
        return current