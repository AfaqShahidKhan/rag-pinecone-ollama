"""
src/infrastructure/pre_processing/unicode_normalizer.py

IPreProcessor adapter that applies Unicode standardization so that
semantically identical text has a consistent byte representation:

  - NFC normalization (compose decomposed characters: e + ́ → é)
  - Smart / curly quotes → straight ASCII quotes
  - Em/en dashes → ASCII hyphens
  - Ellipsis character → three dots
  - BOM marker removal (U+FEFF at start)

This runs after TextSanitizer and before MetadataNormalizer.
"""

from __future__ import annotations

import unicodedata

from src.domain.entities import Document
from src.domain.interfaces import ILogger, IPreProcessor

# Map of typographic Unicode characters → plain ASCII equivalents
_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK
    "\u201a": "'",   # SINGLE LOW-9 QUOTATION MARK
    "\u201c": '"',   # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',   # RIGHT DOUBLE QUOTATION MARK
    "\u201e": '"',   # DOUBLE LOW-9 QUOTATION MARK
    "\u2013": "-",   # EN DASH
    "\u2014": "-",   # EM DASH
    "\u2015": "-",   # HORIZONTAL BAR
    "\u2026": "...", # HORIZONTAL ELLIPSIS
    "\u00b7": "·",   # MIDDLE DOT (keep as-is, already ASCII-safe)
    "\ufeff": "",    # BOM — remove entirely
    "\u00a0": " ",   # NON-BREAKING SPACE → regular space
}

_TRANSLATION_TABLE = str.maketrans(_UNICODE_REPLACEMENTS)


class UnicodeNormalizer(IPreProcessor):
    def __init__(self, logger: ILogger, form: str = "NFC") -> None:
        """
        Args:
            form: Unicode normalization form. 'NFC' (default) composes characters.
                  Use 'NFKC' for compatibility normalization (e.g. ﬁ → fi).
        """
        self._logger = logger
        self._form = form

    def process(self, document: Document) -> Document:
        normalized = self._normalize(document.page_content)
        return Document(page_content=normalized, metadata=document.metadata)

    def _normalize(self, text: str) -> str:
        # 1. Apply Unicode normalization form (NFC by default)
        text = unicodedata.normalize(self._form, text)

        # 2. Replace typographic characters with ASCII equivalents
        text = text.translate(_TRANSLATION_TABLE)

        return text