"""
src/domain/interfaces/extraction_strategy.py

Ports for pluggable, orderable extraction strategies (PDF text, PDF
tables, OCR engines). Each strategy does one job for one page; the
FallbackChainExecutor tries them in configured order until one meets the
confidence threshold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities import ExtractionAttempt


class ITextExtractionStrategy(ABC):
    name: str

    @abstractmethod
    def extract_page(self, path: Path, page_index: int, total_pages: int) -> ExtractionAttempt:
        """Extract text from one page (0-indexed) of a PDF."""
        ...


class ITableExtractionStrategy(ABC):
    name: str

    @abstractmethod
    def extract_page(self, path: Path, page_index: int, total_pages: int) -> ExtractionAttempt:
        """Extract tables (as joined Markdown) from one page (0-indexed) of a PDF."""
        ...


class IOcrEngine(ABC):
    name: str

    @abstractmethod
    def recognize(self, image_path: Path) -> ExtractionAttempt:
        """Run OCR on a rasterized page image, return recognized text."""
        ...