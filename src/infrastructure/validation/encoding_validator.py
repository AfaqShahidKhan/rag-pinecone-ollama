"""
src/infrastructure/validation/encoding_validator.py

Flags content with an excessive ratio of Unicode replacement characters
(U+FFFD) — the character Python substitutes when it can't decode bytes
correctly. A high ratio is a strong signal of an encoding mismatch or file
corruption, even when loading didn't raise an exception.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.entities import Document
from src.domain.errors import ErrorCode, PipelineError
from src.domain.interfaces import IContentValidator

_REPLACEMENT_CHAR = "\ufffd"


class EncodingValidator(IContentValidator):
    def __init__(self, max_replacement_ratio: float = 0.01) -> None:
        self._max_ratio = max_replacement_ratio

    def validate(self, path: Path, documents: list[Document]) -> None:
        for doc in documents:
            text = doc.page_content
            if not text:
                continue
            ratio = text.count(_REPLACEMENT_CHAR) / len(text)
            if ratio > self._max_ratio:
                page = doc.metadata.get("page", "?")
                raise PipelineError(
                    ErrorCode.VALID_ENCODING_ERROR,
                    f"Page {page} is {ratio:.1%} replacement characters — "
                    f"likely an encoding mismatch or file corruption.",
                    file_path=str(path),
                )