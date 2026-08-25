"""
src/infrastructure/extraction/pymupdf_table_strategy.py

Fallback PDF table extraction strategy, used when pdfplumber finds
nothing (e.g. borderless tables pdfplumber's line-detection misses).
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from src.domain.entities import ExtractionAttempt
from src.domain.interfaces import ITableExtractionStrategy
from src.infrastructure.extraction.table_markdown import table_to_markdown


class PyMuPdfTableExtractionStrategy(ITableExtractionStrategy):
    name = "pymupdf_tables"

    def __init__(self) -> None:
        self._cached_path: Path | None = None
        self._cached_doc: fitz.Document | None = None

    def _doc_for(self, path: Path) -> fitz.Document:
        if self._cached_path != path:
            if self._cached_doc is not None:
                self._cached_doc.close()
            self._cached_doc = fitz.open(str(path))
            self._cached_path = path
        return self._cached_doc

    def extract_page(self, path: Path, page_index: int, total_pages: int) -> ExtractionAttempt:
        doc = self._doc_for(path)
        page = doc[page_index]

        markdown_tables: list[str] = []
        try:
            for table in page.find_tables():
                md = table_to_markdown(table.extract())
                if md:
                    markdown_tables.append(md)
        except Exception:
            pass  # treated as "no tables found" — the chain moves on

        content = "\n\n".join(markdown_tables)
        return ExtractionAttempt(
            content=content,
            confidence=1.0 if markdown_tables else 0.0,
            strategy_name=self.name,
        )