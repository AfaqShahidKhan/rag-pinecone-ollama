"""
src/infrastructure/pre_processing/schema_mapper.py

IPreProcessor adapter that acts as the final gate before chunking.
It validates that a Document conforms to the canonical internal schema
and raises ValueError for any document that cannot be salvaged:

  - page_content must be a non-empty string after stripping
  - All required metadata keys must be present (MetadataNormalizer runs first)
  - page_content minimum length is configurable (default: 20 chars)

Documents that fail validation are NOT silently dropped here —
SchemaMapper raises ValueError so PreProcessingPipeline can log and
exclude them with full visibility.

This is deliberately the last processor in the chain so that all
upstream cleaners have already run before the final gate check.
"""

from __future__ import annotations

from src.domain.entities import Document
from src.domain.interfaces import ILogger, IPreProcessor

_REQUIRED_METADATA_KEYS = (
    "source",
    "page",
    "total_pages",
    "file_type",
    "ingested_at",
)


class SchemaMapper(IPreProcessor):
    def __init__(self, logger: ILogger, min_content_length: int = 20) -> None:
        """
        Args:
            min_content_length: Documents shorter than this (after stripping)
                                are considered empty and rejected.
        """
        self._logger = logger
        self._min_length = min_content_length

    def process(self, document: Document) -> Document:
        self._validate_content(document)
        self._validate_metadata(document)

        # Return a clean copy with stripped content (removes any leading/trailing whitespace)
        return Document(
            page_content=document.page_content.strip(),
            metadata=document.metadata,
        )

    def _validate_content(self, document: Document) -> None:
        content = document.page_content
        source = document.metadata.get("source", "?")
        page = document.metadata.get("page", "?")

        if not isinstance(content, str):
            raise ValueError(
                f"[{source} p{page}] page_content must be str, "
                f"got {type(content).__name__}"
            )

        if len(content.strip()) < self._min_length:
            raise ValueError(
                f"[{source} p{page}] page_content too short "
                f"({len(content.strip())} chars, minimum {self._min_length})"
            )

    def _validate_metadata(self, document: Document) -> None:
        source = document.metadata.get("source", "?")
        page = document.metadata.get("page", "?")
        missing = [
            key for key in _REQUIRED_METADATA_KEYS
            if key not in document.metadata
        ]
        if missing:
            raise ValueError(
                f"[{source} p{page}] missing required metadata keys: {missing}. "
                "Ensure MetadataNormalizer runs before SchemaMapper."
            )