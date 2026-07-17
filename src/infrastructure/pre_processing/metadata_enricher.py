"""
src/infrastructure/pre_processing/metadata_enricher.py

IPreProcessor adapter that enriches every Document's metadata dict with
computed fields derived from its content. Runs after SchemaMapper in the
pre-processing pipeline so all required fields are already guaranteed.

Fields added:
  word_count                 int   — number of whitespace-separated tokens
  char_count                 int   — total character count
  estimated_read_time_seconds int  — word_count ÷ 4  (≈ 240 wpm reading speed)
  has_numbers                bool  — True if 3+ consecutive digits appear (e.g. financial data)
  has_tables                 bool  — True if Markdown table syntax is detected ( | --- | )
  content_density            str   — "sparse" | "normal" | "dense" based on word/char ratio

These fields are stored in the vector store metadata and are available for
filtering, debugging, and future routing decisions (Phase 3 DocumentRouter
can inspect them to choose a chunking strategy).
"""

from __future__ import annotations

import re

from src.domain.entities import Document
from src.domain.interfaces import ILogger, IPreProcessor

# Matches sequences of 3+ digits — indicates financial / numerical content
_NUMBERS_RE = re.compile(r"\d{3,}")

# Minimal Markdown table indicator: a row of pipe-separated dashes
_TABLE_RE = re.compile(r"\|[\s\-]+\|")

# Words per minute for estimated reading time
_READING_WPM = 240


class MetadataEnricher(IPreProcessor):
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def process(self, document: Document) -> Document:
        text = document.page_content
        meta = dict(document.metadata)

        word_count = len(text.split())
        char_count = len(text)
        read_time = max(1, round(word_count / (_READING_WPM / 60)))  # seconds

        meta["word_count"] = word_count
        meta["char_count"] = char_count
        meta["estimated_read_time_seconds"] = read_time
        meta["has_numbers"] = bool(_NUMBERS_RE.search(text))
        meta["has_tables"] = bool(_TABLE_RE.search(text))
        meta["content_density"] = self._compute_density(word_count, char_count)

        self._logger.debug(
            f"MetadataEnricher: '{meta.get('source', '?')}' p{meta.get('page', '?')} "
            f"— {word_count} words, {char_count} chars, density={meta['content_density']}"
        )

        return Document(page_content=text, metadata=meta)

    @staticmethod
    def _compute_density(word_count: int, char_count: int) -> str:
        """
        Average English word is ~5 chars. Ratio of char_count/word_count tells
        us whether the content is abbreviation-heavy (sparse), normal prose, or
        dense technical text with long compound words / URLs.
        """
        if word_count == 0:
            return "sparse"
        ratio = char_count / word_count
        if ratio < 4:
            return "sparse"
        if ratio > 8:
            return "dense"
        return "normal"