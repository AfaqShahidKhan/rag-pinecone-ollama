"""
src/infrastructure/extraction/pypdf_text_strategy.py

Primary PDF text extraction strategy. Caches the opened PdfReader per path
so processing a multi-hundred-page document only opens the file once, not
once per page.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from src.domain.confidence import score_text_confidence
from src.domain.entities import ExtractionAttempt
from src.domain.interfaces import ITextExtractionStrategy


class PypdfTextExtractionStrategy(ITextExtractionStrategy):
    name = "pypdf"

    def __init__(self) -> None:
        self._cached_path: Path | None = None
        self._cached_reader: PdfReader | None = None

    def _reader_for(self, path: Path) -> PdfReader:
        if self._cached_path != path:
            self._cached_reader = PdfReader(str(path))
            self._cached_path = path
        return self._cached_reader

    def extract_page(self, path: Path, page_index: int, total_pages: int) -> ExtractionAttempt:
        reader = self._reader_for(path)
        text = (reader.pages[page_index].extract_text() or "").strip()
        return ExtractionAttempt(
            content=text,
            confidence=score_text_confidence(text),
            strategy_name=self.name,
        )