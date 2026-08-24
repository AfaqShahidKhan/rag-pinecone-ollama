"""
src/infrastructure/validation/content_not_empty_validator.py

Rejects files that loaded "successfully" (no exception) but produced no
usable text at all across every page/section — e.g. a scanned PDF with no
extractable text. A loader silently returning zero content is treated the
same as a hard failure.
"""

from __future__ import annotations

from pathlib import Path

from src.domain.entities import Document
from src.domain.errors import ErrorCode, PipelineError
from src.domain.interfaces import IContentValidator


class ContentNotEmptyValidator(IContentValidator):
    def validate(self, path: Path, documents: list[Document]) -> None:
        total_chars = sum(len(doc.page_content.strip()) for doc in documents)
        if total_chars == 0:
            raise PipelineError(
                ErrorCode.VALID_EMPTY_CONTENT,
                "Loader produced no usable text content across any page/section.",
                file_path=str(path),
            )