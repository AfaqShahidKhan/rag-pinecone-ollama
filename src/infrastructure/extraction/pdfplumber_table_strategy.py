"""
src/infrastructure/extraction/pdfplumber_table_strategy.py

Primary PDF table extraction strategy (moved from the old PdfDocumentLoader
implementation, unchanged in behavior — now just reusable as a strategy).
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from src.domain.entities import ExtractionAttempt
from src.domain.interfaces import ITableExtractionStrategy
from src.infrastructure.extraction.table_markdown import table_to_markdown


class PdfplumberTableExtractionStrategy(ITableExtractionStrategy):
    name = "pdfplumber"

    def __init__(self) -> None:
        self._cached_path: Path | None = None
        self._cached_pdf: pdfplumber.PDF | None = None

    def _pdf_for(self, path: Path) -> pdfplumber.PDF:
        if self._cached_path != path:
            if self._cached_pdf is not None:
                self._cached_pdf.close()
            self._cached_pdf = pdfplumber.open(str(path))
            self._cached_path = path
        return self._cached_pdf

    def extract_page(self, path: Path, page_index: int, total_pages: int) -> ExtractionAttempt:
        pdf = self._pdf_for(path)
        page = pdf.pages[page_index]
        raw_tables = page.extract_tables() or []
        markdown_tables = [md for md in (table_to_markdown(t) for t in raw_tables) if md]
        content = "\n\n".join(markdown_tables)
        return ExtractionAttempt(
            content=content,
            confidence=1.0 if markdown_tables else 0.0,
            strategy_name=self.name,
        )