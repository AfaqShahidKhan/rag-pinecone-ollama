"""
src/infrastructure/extraction/null_table_strategy.py

Ultimate table-extraction fallback: give up on structured Markdown tables.
The plain text extraction (pypdf/PyMuPDF/OCR) already contains the table's
cell values inline, just not as a structured table — so this always
"succeeds" with empty content, ensuring the chain always terminates
cleanly instead of the document ending up with zero table content AND an
error.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.entities import ExtractionAttempt
from src.domain.interfaces import ITableExtractionStrategy


class NullTableExtractionStrategy(ITableExtractionStrategy):
    name = "text_extraction"

    def extract_page(self, path: Path, page_index: int, total_pages: int) -> ExtractionAttempt:
        return ExtractionAttempt(content="", confidence=1.0, strategy_name=self.name)