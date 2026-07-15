"""
src/infrastructure/pre_processing/metadata_normalizer.py

IPreProcessor adapter that standardizes the metadata dict on every Document:

  - Ensures all required keys exist (with safe defaults if missing)
  - Normalises file_type to lowercase
  - Normalises source to just the filename (strips full path)
  - Adds ingested_at timestamp in ISO 8601 format (UTC)
  - Coerces page / total_pages / chunk_index / chunk_total to int

This makes downstream components (vector store metadata, eval reporter,
Streamlit source display) safe to rely on consistent field names and types.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.domain.entities import Document
from src.domain.interfaces import ILogger, IPreProcessor

# Keys that every Document metadata dict must contain after normalization.
_REQUIRED_KEYS: dict[str, object] = {
    "source":      "unknown",
    "source_path": "",
    "page":        0,
    "total_pages": 0,
    "chunk_index": 0,
    "chunk_total": 0,
    "file_type":   "unknown",
}


class MetadataNormalizer(IPreProcessor):
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def process(self, document: Document) -> Document:
        meta = self._normalize(dict(document.metadata))
        return Document(page_content=document.page_content, metadata=meta)

    def _normalize(self, meta: dict) -> dict:
        # 1. Fill in missing required keys with safe defaults
        for key, default in _REQUIRED_KEYS.items():
            if key not in meta:
                meta[key] = default

        # 2. Normalize source → filename only (strip full path)
        raw_source = str(meta["source"])
        meta["source"] = Path(raw_source).name if raw_source else "unknown"

        # 3. Normalize file_type to lowercase string
        meta["file_type"] = str(meta.get("file_type", "unknown")).lower().strip()

        # 4. Coerce numeric fields to int (loaders may leave them as str)
        for int_field in ("page", "total_pages", "chunk_index", "chunk_total"):
            try:
                meta[int_field] = int(meta[int_field])
            except (ValueError, TypeError):
                meta[int_field] = 0

        # 5. Add ingested_at timestamp (ISO 8601 UTC) — idempotent: never overwrite
        if "ingested_at" not in meta:
            meta["ingested_at"] = datetime.now(timezone.utc).isoformat()

        return meta