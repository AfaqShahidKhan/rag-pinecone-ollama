"""
src/infrastructure/loaders/json_loader.py

IDocumentLoader adapter for JSON files. Uses stdlib json — no extra install.

Handles three common JSON shapes:
  1. Array of objects  → one Document per object (e.g. exported Q&A datasets)
  2. Single object     → one Document (all string values concatenated)
  3. Array of strings  → one Document per string entry

For objects, string values are extracted recursively so nested structures
(e.g. {"title": "...", "body": {"intro": "...", "detail": "..."}})
are handled correctly. Non-string leaf values (numbers, booleans) are
included as "key: value" pairs so no information is silently discarded.

Keys listed in SKIP_KEYS are excluded from text extraction — extend this
list for your data shape (e.g. add "id", "timestamp", "url" if irrelevant).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.domain.entities import Document
from src.domain.interfaces import IDocumentLoader, ILogger

SUPPORTED_EXTENSION = ".json"

# Metadata-only keys that should not be included in page_content
SKIP_KEYS: frozenset[str] = frozenset({
    "id", "_id", "uuid", "created_at", "updated_at",
    "timestamp", "date", "url", "href", "link",
})


class JsonLoader(IDocumentLoader):
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == SUPPORTED_EXTENSION

    def load(self, path: Path) -> list[Document]:
        raw = path.read_text(encoding="utf-8", errors="replace")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._logger.error(f"Failed to parse JSON '{path.name}': {exc}")
            return []

        docs = self._extract_documents(data, path)

        if not docs:
            self._logger.warning(f"'{path.name}' yielded no text after parsing — skipping.")
            return []

        self._logger.info(f"Loaded '{path.name}': {len(docs)} document(s).")
        return docs

    def _extract_documents(self, data: object, path: Path) -> list[Document]:
        if isinstance(data, list):
            return self._load_array(data, path)
        if isinstance(data, dict):
            return self._load_single_object(data, path, index=1, total=1)
        # Scalar root (unusual but valid JSON)
        text = str(data).strip()
        return [self._make_document(text, path, 1, 1)] if text else []

    def _load_array(self, items: list, path: Path) -> list[Document]:
        docs: list[Document] = []
        total = len(items)

        for i, item in enumerate(items):
            if isinstance(item, str):
                text = item.strip()
                if text:
                    docs.append(self._make_document(text, path, i + 1, total))
            elif isinstance(item, dict):
                item_docs = self._load_single_object(item, path, i + 1, total)
                docs.extend(item_docs)

        return docs

    def _load_single_object(
        self, obj: dict, path: Path, index: int, total: int
    ) -> list[Document]:
        parts = self._flatten_object(obj)
        text = "\n".join(parts).strip()
        if not text:
            return []
        return [self._make_document(text, path, index, total)]

    def _flatten_object(self, obj: dict, prefix: str = "") -> list[str]:
        """Recursively extract key: value pairs as text lines."""
        lines: list[str] = []
        for key, value in obj.items():
            if key in SKIP_KEYS:
                continue
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                lines.extend(self._flatten_object(value, prefix=full_key))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        lines.append(f"{full_key}: {item.strip()}")
                    elif isinstance(item, dict):
                        lines.extend(self._flatten_object(item, prefix=full_key))
            elif isinstance(value, str) and value.strip():
                lines.append(f"{full_key}: {value.strip()}")
            elif value is not None and not isinstance(value, bool):
                lines.append(f"{full_key}: {value}")
        return lines

    @staticmethod
    def _make_document(text: str, path: Path, index: int, total: int) -> Document:
        return Document(
            page_content=text,
            metadata={
                "source":      path.name,
                "source_path": str(path),
                "page":        index,
                "total_pages": total,
                "file_type":   "json",
            },
        )