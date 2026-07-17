"""
src/infrastructure/pre_processing/text_sanitizer.py

IPreProcessor adapter that removes low-level text noise before any
further normalization or chunking occurs:

  - Null bytes and soft hyphens (common PDF extraction artifacts)
  - Non-printable ASCII control characters (except \n \r \t)
  - Repeated paragraph separators and trailing whitespace per line
  - Zero-width and other invisible Unicode characters

Deliberately does NOT touch whitespace collapsing or spaced-character
repair — those are handled downstream in RecursiveTextChunker so we
avoid double-processing.
"""

from __future__ import annotations

import re
import unicodedata

from src.domain.entities import Document
from src.domain.interfaces import ILogger, IPreProcessor

# Control chars to strip: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F
# Keeps: 0x09 (tab), 0x0A (LF), 0x0D (CR)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Soft hyphen (U+00AD) — invisible but causes tokenisation issues
_SOFT_HYPHEN = "\u00ad"

# Zero-width characters
_ZERO_WIDTH = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00a0]"
)

# More than 3 consecutive blank lines → 2 blank lines
_EXCESSIVE_BLANK_LINES_RE = re.compile(r"\n{4,}")


class TextSanitizer(IPreProcessor):
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger

    def process(self, document: Document) -> Document:
        original_len = len(document.page_content)
        cleaned = self._sanitize(document.page_content)
        delta = original_len - len(cleaned)

        if delta > 0:
            self._logger.debug(
                f"TextSanitizer: removed {delta} chars from "
                f"'{document.metadata.get('source', '?')}' "
                f"p{document.metadata.get('page', '?')}"
            )

        return Document(page_content=cleaned, metadata=document.metadata)

    @staticmethod
    def _sanitize(text: str) -> str:
        # 1. Remove null bytes and soft hyphens
        text = text.replace("\x00", "").replace(_SOFT_HYPHEN, "")

        # 2. Strip other control characters
        text = _CONTROL_CHAR_RE.sub("", text)

        # 3. Remove zero-width / invisible Unicode characters
        text = _ZERO_WIDTH.sub("", text)

        # 4. Strip trailing whitespace from each line
        text = "\n".join(line.rstrip() for line in text.splitlines())

        # 5. Collapse excessive blank lines
        text = _EXCESSIVE_BLANK_LINES_RE.sub("\n\n\n", text)

        return text.strip()