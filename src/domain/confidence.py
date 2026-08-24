"""
src/domain/confidence.py

Shared heuristic for scoring extracted text quality — used to decide
whether a fallback extraction strategy should be tried. None of the
underlying libraries (pypdf, pdfplumber, PyMuPDF, OCR engines) report a
trust score themselves, so this computes one from the output.

Heuristic (0.0-1.0):
  - empty/whitespace-only text -> 0.0 immediately
  - readable-character ratio (letters/digits/common punctuation vs. total)
  - average "word" length sanity — extremely short/long "words" suggest
    garbled extraction (spaced-out characters, binary noise)
"""

from __future__ import annotations

import re

_READABLE_CHARS_RE = re.compile(r"[A-Za-z0-9\s.,;:!?'\"()\-]")
_WORD_RE = re.compile(r"\S+")


def score_text_confidence(text: str) -> float:
    if not text or not text.strip():
        return 0.0

    total_chars = len(text)
    readable_chars = len(_READABLE_CHARS_RE.findall(text))
    readable_ratio = readable_chars / total_chars

    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / len(words)
    # Sane average word length for real prose is roughly 2-12 chars.
    # Penalize far outside that range (garbled/binary noise or one giant blob).
    word_len_score = 1.0 if 2 <= avg_word_len <= 12 else 0.5

    return round(min(readable_ratio, 1.0) * word_len_score, 3)