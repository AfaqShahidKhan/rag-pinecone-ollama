"""
src/infrastructure/extraction/pymupdf_text_strategy.py

First fallback for PDF text extraction. PyMuPDF handles some layouts
pypdf struggles with (certain embedded fonts, rotated text).
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from src.domain.confidence import score_text_confidence
from src.domain.entities import ExtractionAttempt
from src.domain.interfaces import ITextExtractionStrategy


class PyMuPdfTextExtractionStrategy(ITextExtractionStrategy):
    name = "pymupdf"

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
        text = doc[page_index].get_text().strip()
        return ExtractionAttempt(
            content=text,
            confidence=score_text_confidence(text),
            strategy_name=self.name,
        )